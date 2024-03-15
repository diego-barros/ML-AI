from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    #log the training loss at every iteration, the training accuracy at each epoch and 
    #the validation accuracy at each epoch. Log everything in global training steps. 

    # global step variable
    global_step = 0

    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        #traning accuracy at each epoch
        training_accuracy = 0
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)

            # need to extract the scalar, otherwise getting error AssertionError: Tensor should contain one element 
            #(0 dimensions). Was given size: 10 and 1 dimensions.
            dummy_train_accuracy = (epoch / 10. + torch.randn(10).mean()).item()
           
            # training accuracy
            training_accuracy = training_accuracy + dummy_train_accuracy

           
            # log the training loss - logger.add_scalar('train/loss', t_loss, 0)
            train_logger.add_scalar('loss', dummy_train_loss, global_step)

            # increment global step
            global_step = global_step + 1

         # divide by number of iterations to get the average value
        average_training_accuracy = training_accuracy / 20
        
        train_logger.add_scalar('accuracy', average_training_accuracy, global_step - 1)


        torch.manual_seed(epoch)

        # need another var to store the values of dummy_validation_accuracy through 10 iterations
        accuracy_sum = 0

        for iteration in range(10):
            dummy_validation_accuracy = (epoch / 10. + torch.randn(10).mean()).item()
            accuracy_sum = accuracy_sum + dummy_validation_accuracy

        # divide by number of iterations to get the average value
        average_validation_accuracy = accuracy_sum / 10

        valid_logger.add_scalar('accuracy', average_validation_accuracy, global_step - 1)



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
