class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            # foler that contains leftImg8bit/
            return r'/home/common_user/danet-pytorch/Cityspace'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError("undefined dataset {}.".format(dataset))
