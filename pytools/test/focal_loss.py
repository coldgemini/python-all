def dice_loss_fun(self, logits, input_gt):
    input_gt = tf.one_hot(input_gt, self.num_classes)
    # print(input_gt.shape)
    # pred = logits
    pred = tf.nn.softmax(logits)
    smooth = 1.0
    dice = 0
    # print("dice_loss_fun shape")
    # print(input_gt.shape)
    # print(pred.shape)
    for i in range(self.num_classes):
        inse = tf.reduce_sum(pred[:, :, :, :, i] * input_gt[:, :, :, :, i])
        # l = tf.reduce_sum(pred[:, :, :, :, i] * pred[:, :, :, :, i])
        l = tf.reduce_sum(pred[:, :, :, :, i])
        # r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
        r = tf.reduce_sum(input_gt[:, :, :, :, i])
        dice = dice + (2 * inse + smooth) / (l + r + smooth)
        # dice = dice + (2 * inse) / (l + r)
    return -dice
