import tensorflow as tf

x1 = tf.constant(5)  # since it is constant the tensor will not change
x2 = tf.constant(6)


result = tf.multiply(x1,x2)

print(result)  # this is just an abstract tensor. It won't give you the answer.


'''sess = tf.Session()
print(sess.run(result))  # until we run the session, no process was executed, nothing happened.
sess.close()'''

with tf.Session() as sess:  # This will automatically close your Session,
    # without actually specifying the close statement.
    print(sess.run(result))

