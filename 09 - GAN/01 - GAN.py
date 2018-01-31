# 2016년에 가장 관심을 많이 받았던 비감독(Unsupervised) 학습 방법인
# Generative Adversarial Network(GAN)을 구현해봅니다.
# https://arxiv.org/abs/1406.2661

#########
# 기본적인import,데이터가져오기
######

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 옵션 설정
######
total_epoch = 100      # 반복 수행 횟수
batch_size = 100       # 한번에 만드는 이미지 갯수
learning_rate = 0.0002 # 학습 시 변수를 조정하는 정도
# 신경망 레이어 구성 옵션
n_hidden = 256         # 신경망(두뇌)의 복잡도
n_input = 28 * 28      # 입력이미지의 크기
n_noise = 128          # 생성기의 입력값으로 사용할 노이즈의 크기

#########
# 신경망 모델 구성
######
################################모델 생성 시작################################################
################################공장이라고보면=>생산기계를 만드는 과정################################################
# 입력 이미지 X
X = tf.placeholder(tf.float32, [None, n_input])
# 노이즈 Z를 입력값으로 사용합니다.
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성기 신경망에 사용하는 변수들입니다.
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 판별기 신경망에 사용하는 변수들입니다.
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# 판별기의 최종 결과값은 얼마나 진짜와 가깝냐를 판단하는 단 하나의 값
# 판별기에서 열심히 계산한 결과를 단 하나의 값으로 만들기 위해 아래의 연산을 사용
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


# 생성기(G) 신경망을 구성합니다.
def generator(noise_z):
    hidden = tf.nn.relu(
                    tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, G_W2) + G_b2)

    return output


# 판별기(D) 신경망을 구성합니다.
def discriminator(inputs):
    hidden = tf.nn.relu(
                    tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, D_W2) + D_b2)

    return output


# 랜덤한 노이즈(Z)를 만듭니다.
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

################################모델 생성 완료################################################


################################학습 시나리오 작성 시작################################################
################################공장이라고보면=>생산기계를 배치하는 과정################################################
# 1. (생성자)노이즈를 이용해 랜덤한 이미지를 생성합니다.
G = generator(Z)

################################공장이라고보면=>작업하는 룰을 결정하는 과정################################################
# 2-1. (판별자)노이즈를 이용해 생성한 이미지가 진짜 이미지인지 판별한 값을 구합니다.
D_gene = discriminator(G)      # 가짜이미지 판별 => 0일수록 좋음
# 2-2. (판별자)진짜 이미지를 이용해 판별한 값을 구합니다.
D_real = discriminator(X)      # 진짜이미지 판별 => 1일수록 좋음
# 2-3. (판별자) D_gene(가짜이미지 판별)은 최대한작게, D_real(진짜이미지 판별)은 최대한크게 만들도록 내부변수를 조정
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))


# (생성자) D_gene(가짜이미지 판별)을 최대한크게하도록 내부변수를 조정
loss_G = tf.reduce_mean(tf.log(D_gene))

# loss_D 를 구할 때는 판별기 신경망에 사용되는 변수만 사용하고,
# loss_G 를 구할 때는 생성기 신경망에 사용되는 변수만 사용하여 최적화를 합니다.
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

########## <공장이라고생각하면=>이전 결과를 토대로 작업방식을 최적화하는 방법을 정함> ##########
# GAN 논문의 수식에 따르면 loss 를 극대화 해야하지만, minimize 하는 최적화 함수를 사용하기 때문에
# 최적화 하려는 loss_D 와 loss_G 에 음수 부호를 붙여줍니다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                                                         var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,
                                                         var_list=G_var_list)

################################학습 시나리오 작성 완료################################################

#########
# 신경망 모델 학습
######

################################학습 시나리오 실행 시작################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

########## <공장이라고생각하면=>전기와 원재료를 넣어서 공장을 가동함> ##########
for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        # 판별기와 생성기 신경망을 각각 학습시킵니다.
        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

    #########
    # 학습이 되어가는 모습을 보기 위해 주기적으로 이미지를 생성하여 저장
    ######
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료!')
################################학습 시나리오 실행 완료################################################
