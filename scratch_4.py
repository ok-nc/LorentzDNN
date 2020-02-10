import datareader
import flagreader
import os

flags = flagreader.read_flag()

train_loader, test_loader = datareader.read_data(x_range=flags.x_range,
                                                      y_range=flags.y_range,                                                      geoboundary=flags.geoboundary,
                                                      batch_size=flags.batch_size,
                                                      normalize_input=flags.normalize_input,
                                                      data_dir=flags.data_dir,
                                                      test_ratio=flags.test_ratio)


