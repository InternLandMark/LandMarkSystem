# 训练API
介绍训练相关的api，trainer主要API
### NeRFTrainer
class NeRFTrainer(config)
这个类是Trainer的基类，实例化需要提供一个config参数，config参数参考上文。config一般由[train_config, dataset_config, model_config]组成，train_config可以由上午TrainConfig实例化，dataset_config可以由DatasetConfig实例化，model_config根据不同的模型由用户提供。

* init_train_env()
init_train_env是Trainer的一个接口，目的是实现根据用户config配置训练环境，通常不需要用户实现。

* create_model()
create_model是一个需要用户实现的接口，用户需要在这个接口里面实现模型创建。

* create_optimizer()
create_optimizer是一个需要用户实现的接口，用户应该在这个接口里面实现优化器的创建。

* check_args()
这个接口用于检查用户config是否有错误，用户可以重载这个接口以实现对应的检查。

* train(*args, **kwargs)
这个接口用于启动训练，用户应该实现启动训练的接口。

* evaluation(*args, **kwargs)
这个接口用于执行evaluation，用户应该实现这个接口实现evaluation功能。
