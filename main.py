import tensorflow as tf
import numpy as np
import helper
from train import train_nn,optimize

num_classes = 2
NUMBER_OF_CLASSES = 2
image_shape = (160, 576)
IMAGE_SHAPE = (160,576)
EPOCHS = 40
BATCH_SIZE = 16
DROPOUT = 0.75

# Specify these directory paths

data_dir = '../data_road'
runs_dir = './runs'
training_dir ='../data_road/training'
vgg_path = '../vgg'

#--------------------------
# PLACEHOLDER TENSORS
#--------------------------

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

#--------------------------
# FUNCTIONS
#--------------------------

def load_vgg(sess, vgg_path):
  
  # load the model and weights
  model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

  # Get Tensors to be returned from graph
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name('image_input:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  layer3 = graph.get_tensor_by_name('layer3_out:0')
  layer4 = graph.get_tensor_by_name('layer4_out:0')
  layer7 = graph.get_tensor_by_name('layer7_out:0')

  return image_input, keep_prob, layer3, layer4, layer7


def conv_1x1(layer, layer_name):
  """ Return the output of a 1x1 convolution of a layer """
  return tf.layers.conv2d(inputs = layer,
                          filters =  NUMBER_OF_CLASSES,
                          kernel_size = (1, 1),
                          strides = (1, 1),
                          name = layer_name)


def upsample(layer, k, s, layer_name):
  """ Return the output of transpose convolution given kernel_size k and strides s """
  return tf.layers.conv2d_transpose(inputs = layer,
                                    filters = NUMBER_OF_CLASSES,
                                    kernel_size = (k, k),
                                    strides = (s, s),
                                    padding = 'same',
                                    name = layer_name)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes = NUMBER_OF_CLASSES):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  vgg_layerX_out: TF Tensor for VGG Layer X output
  num_classes: Number of classes to classify
  return: The Tensor for the last layer of output
  """

  # Use a shorter variable name for simplicity
  layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

  # Apply a 1x1 convolution to encoder layers
  layer3x = conv_1x1(layer = layer3, layer_name = "layer3conv1x1")
  layer4x = conv_1x1(layer = layer4, layer_name = "layer4conv1x1")
  layer7x = conv_1x1(layer = layer7, layer_name = "layer7conv1x1")
 
  # Add decoder layers to the network with skip connections and upsampling
  # Note: the kernel size and strides are the same as the example in Udacity Lectures
  #       Semantic Segmentation Scene Understanding Lesson 10-9: FCN-8 - Decoder
  decoderlayer1 = upsample(layer = layer7x, k = 4, s = 2, layer_name = "decoderlayer1")
  decoderlayer2 = tf.add(decoderlayer1, 0.5*layer4x, name = "decoderlayer2")
  decoderlayer3 = upsample(layer = decoderlayer2, k = 4, s = 2, layer_name = "decoderlayer3")
  decoderlayer4 = tf.add(decoderlayer3, 1.5*layer3x, name = "decoderlayer4")
  decoderlayer_output = upsample(layer = decoderlayer4, k = 16, s = 8, layer_name = "decoderlayer_output")

  return decoderlayer_output


# def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
   
#     # Use a shorter variable name for simplicity
#     layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

#     # Apply 1x1 convolution in place of fully connected layer
#     fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

#     # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
#     fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
#     kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

#     # Add a skip connection between current final layer fcn8 and 4th layer
#     fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

#     # Upsample again
#     fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
#     kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

#     # Add skip connection
#     fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

#     # Upsample again
#     fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
#     kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

#     return fcn11




def run():
  
  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)

  # A function to get batches
  get_batches_fn = helper.gen_batch_function(training_dir, image_shape)
  
  with tf.Session() as session:
        
    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)

    # The resulting network architecture from adding a decoder on top of the given vgg model
    model_output = layers(layer3, layer4, layer7, num_classes)

    # Returns the output logits, training operation and cost operation to be used
    # - logits: each row represents a pixel, each column a class
    # - train_op: function used to get the right parameters to the model to correctly label the pixels
    # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)
    
    # Initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    print("Model build successful, starting training")

    # Train the neural network
    train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, 
             train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

    # Run the model with the test images and save each painted output image (roads painted green)
    helper.save_inference_samples(runs_dir, data_dir, session, image_shape, logits, keep_prob, image_input)
    
    print("Done!")

#--------------------------
# MAIN
#--------------------------
if __name__ == '__main__':
    run()
