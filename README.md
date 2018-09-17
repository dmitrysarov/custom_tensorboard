tb = TrainValTensorBoard(log_dir = log_dir)

model.fit_generator(data_generator(train_set), 
              steps_per_epoch=train_num//BATCH_SIZE, 
              epochs=100, 
              verbose=1, 
              callbacks=[tb, chpt], 
              validation_data=data_generator(val_set), 
              validation_steps=val_num//BATCH_SIZE,
              class_weight='None',
              max_queue_size=10, workers=1, use_multiprocessing=False, 
              shuffle=False, initial_epoch=0)
