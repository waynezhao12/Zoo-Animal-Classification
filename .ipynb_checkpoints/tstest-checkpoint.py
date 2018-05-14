{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'tensorflow'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-abdf435293ab>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mos\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mos\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0menviron\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'TF_CPP_MIN_LOG_LEVEL'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'2'\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mtensorflow\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'tensorflow'"
     ]
    }
   ],
   "source": [
    "import os\n",
    "os.environ['TF_CPP_MIN_LOG_LEVEL']='2'\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "\n",
    "dataset = pd.read_csv('../input/zoo.csv', header=0)\n",
    "dataset = pd.get_dummies(dataset, columns=['animal_name'])\n",
    "\n",
    "values = list(dataset.columns.values)\n",
    "Y = dataset[values[-100:]]\n",
    "Y = np.array(Y, dtype=np.float32)\n",
    "X = dataset[values[0:-100]]\n",
    "X = np.array(X, dtype=np.float32)\n",
    "# Session\n",
    "sess = tf.Session()\n",
    "# Interval / Epochs\n",
    "interval = 100\n",
    "epoch = 1500\n",
    "\n",
    "\n",
    "#Initialize Neural Network\n",
    "X_data = tf.placeholder(dtype=np.float32, shape=[None, 17])\n",
    "Y_target = tf.placeholder(dtype=np.float32, shape=[None, 100])\n",
    "\n",
    "hidden_layer_nodes = 16\n",
    "\n",
    "w1 = tf.Variable(tf.random_normal(shape=[17, hidden_layer_nodes]))\n",
    "b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))\n",
    "w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 100]))\n",
    "b2 = tf.Variable(tf.random_normal(shape=[100]))\n",
    "\n",
    "hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))\n",
    "final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))\n",
    "\n",
    "#loss = tf.reduce_mean(-tf.reduce_sum(Y_target * tf.log(final_output), axis=0))\n",
    "loss = tf.reduce_mean(-tf.reduce_sum(Y_target * tf.log(final_output + 1e-10), axis=0))\n",
    "\n",
    "# Optimizer\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)\n",
    "#optimizer = tf.train.AdamOptimizer().minimize(loss)\n",
    "\n",
    "# Initialize variables\n",
    "init = tf.global_variables_initializer()\n",
    "sess.run(init)\n",
    "# Training\n",
    "print('Training the model...')\n",
    "for i in range(1, (epoch + 1)):\n",
    "    sess.run(optimizer, feed_dict={X_data: X, Y_target: Y})\n",
    "    if i % interval == 0:\n",
    "        print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X, Y_target: Y}))\n",
    "# Prediction\n",
    "print(\"\\nTrying to predict Buffolo (Index 6) ...\")\n",
    "flower = np.array([[1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1,1]], np.float32)\n",
    "print(np.rint(sess.run(final_output, feed_dict={X_data: flower})))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
