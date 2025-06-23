import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_2 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_3 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_4 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_5 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_6 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_7 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_8 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_9 = np.zeros((self.num_class,) * 2)


    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc


    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc


    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    

# +----------------------------------------------------------------------------------+
    def Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix) / (
                    np.sum(self.adjusted_confusion_matrix, axis=1) + np.sum(self.adjusted_confusion_matrix, axis=0) -
                    np.diag(self.adjusted_confusion_matrix))
        #print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
# +----------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------+
    def Second_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_2) / (
                    np.sum(self.adjusted_confusion_matrix_2, axis=1) + np.sum(self.adjusted_confusion_matrix_2, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_2))
        #print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Third_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_3) / (
                    np.sum(self.adjusted_confusion_matrix_3, axis=1) + np.sum(self.adjusted_confusion_matrix_3, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_3))
        #print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Ninth_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_9) / (
                    np.sum(self.adjusted_confusion_matrix_9, axis=1) + np.sum(self.adjusted_confusion_matrix_9, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_9))
        #print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
# +----------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------+
    def Fourth_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_4) / (
                    np.sum(self.adjusted_confusion_matrix_4, axis=1) + np.sum(self.adjusted_confusion_matrix_4, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_4))
        # print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Fifth_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_5) / (
                    np.sum(self.adjusted_confusion_matrix_5, axis=1) + np.sum(self.adjusted_confusion_matrix_5, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_5))
        # print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Sixth_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_6) / (
                    np.sum(self.adjusted_confusion_matrix_6, axis=1) + np.sum(self.adjusted_confusion_matrix_6, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_6))
        # print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Seventh_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_7) / (
                    np.sum(self.adjusted_confusion_matrix_7, axis=1) + np.sum(self.adjusted_confusion_matrix_7, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_7))
        # print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Eighth_Adjusted_Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.adjusted_confusion_matrix_8) / (
                    np.sum(self.adjusted_confusion_matrix_8, axis=1) + np.sum(self.adjusted_confusion_matrix_8, axis=0) -
                    np.diag(self.adjusted_confusion_matrix_8))
        # print(type(MIoU), MIoU.shape)
        MIoU = np.nanmean(MIoU)
        return MIoU
# +----------------------------------------------------------------------------------+
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    
    def _generate_weighted_matrix(self, gt_image, pre_image, weight_map):
        # print('+-------------------------SOS----------------------------------+')
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # print((self.num_class * gt_image[mask].astype('int')).shape)
        # print(pre_image.shape, gt_image.shape, mask.shape)
        # return self.confusion_matrix
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        weights = weight_map[mask]
        # print(type(label), type(weights))
        # print(label.shape, weights.shape)
        count = np.bincount(label, weights=weights, minlength=self.num_class**2)
        # print(type(count))
        # print(count.size)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, gt_image_0, pre_image_0, visual_loss_map_0):
        assert pre_image_0.shape == gt_image_0.shape
        # print('Attention: {}, {}'.format(type(gt_image), gt_image.shape))
        # print('{}, {}'.format(type(visual_loss_map), visual_loss_map.shape))
        for lp, lt in zip(pre_image_0, gt_image_0):
            self.confusion_matrix += self._generate_matrix(lt.flatten(), lp.flatten())
        
        maps = []
        # 1 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        # print(type(visual_loss_map[0][250][250]), type(pre_image[0][250][250]))
        pre_image_1 = pre_image_0
        gt_image_1 = gt_image_0
        # pre_image_1[pre_image_1 == 0] = -1
        # gt_image_1[gt_image_1 == 0] = -1
        visual_loss_map[visual_loss_map == -255] = 0
        maps.append(visual_loss_map)
        # adjusted_pre_image = np.multiply(pre_image_1, visual_loss_map, dtype=np.int64, casting='unsafe')
        # adjusted_gt_image = np.multiply(gt_image_1, visual_loss_map, dtype=np.int64, casting='unsafe')
        # for lp, lt in zip(adjusted_pre_image, adjusted_gt_image):
        #     lt_flatten = lt.flatten()
        #     lt_flatten = lt_flatten[(lt_flatten > 0) | (lt_flatten == -1)]
        #     lt_flatten[lt_flatten == -1] = 0

        #     lp_flatten = lp.flatten()
        #     lp_flatten = lp_flatten[(lp_flatten > 0) | (lp_flatten == -1)]
        #     lp_flatten[lp_flatten == -1] = 0
        #     #print(lt_flatten.shape, lp_flatten.shape)
        #     self.adjusted_confusion_matrix += self._generate_matrix(lt_flatten, lp_flatten)
        # for lp, lt, wm in zip(pre_image_0, gt_image_0, visual_loss_map):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)


        # 2 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        # print(type(visual_loss_map[0][250][250]), type(pre_image[0][250][250]))
        # pre_image_2 = pre_image_0
        # gt_image_2 = gt_image_0
        # pre_image_2[pre_image_2 == 0] = -1
        # gt_image_2[gt_image_2 == 0] = -1
        # second_visual_loss_map[second_visual_loss_map == 0] = -100
        visual_loss_map[visual_loss_map == 1] = -1
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == -1] = 0
        visual_loss_map[visual_loss_map == -255] = 1
        maps.append(visual_loss_map)

        # 3 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        visual_loss_map[visual_loss_map == 1] = -1
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == -1] = 0
        visual_loss_map[visual_loss_map == -255] = 0
        maps.append(visual_loss_map)

        # 4 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        visual_loss_map[visual_loss_map == 1] = 2
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == -255] = 1
        maps.append(visual_loss_map)

        # 5 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        visual_loss_map[visual_loss_map == 1] = 5
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == -255] = 1
        maps.append(visual_loss_map)

        # 6 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        visual_loss_map[visual_loss_map == 1] = 10
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == -255] = 1
        maps.append(visual_loss_map)

        # 7 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        visual_loss_map[visual_loss_map == 1] = 2
        visual_loss_map[visual_loss_map == 0] = 0
        visual_loss_map[visual_loss_map == -255] = 0
        maps.append(visual_loss_map)

        # 8 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        visual_loss_map[visual_loss_map == 1] = 5
        visual_loss_map[visual_loss_map == 0] = 0
        visual_loss_map[visual_loss_map == -255] = 0
        maps.append(visual_loss_map)

        # 9 map
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        visual_loss_map[visual_loss_map == -255] = 1
        maps.append(visual_loss_map)

        matrices = [self.adjusted_confusion_matrix, self.adjusted_confusion_matrix_2, 
                    self.adjusted_confusion_matrix_3, self.adjusted_confusion_matrix_4, 
                    self.adjusted_confusion_matrix_5, self.adjusted_confusion_matrix_6, 
                    self.adjusted_confusion_matrix_7, self.adjusted_confusion_matrix_8,
                    self.adjusted_confusion_matrix_9]

        for i in range(9):
            for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[i]):
                lt_flatten = lt.flatten()
                lp_flatten = lp.flatten()
                wm_flatten = wm.flatten()
                matrices[i] += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        self.adjusted_confusion_matrix, self.adjusted_confusion_matrix_2, self.adjusted_confusion_matrix_3, self.adjusted_confusion_matrix_4, self.adjusted_confusion_matrix_5, self.adjusted_confusion_matrix_6, self.adjusted_confusion_matrix_7, self.adjusted_confusion_matrix_8, self.adjusted_confusion_matrix_9 = matrices

        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[0]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)

        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[1]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix_2 += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[2]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix_3 += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[3]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix_4 += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[4]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix_5 += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[5]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix_6 += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[6]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix_7 += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        # for lp, lt, wm in zip(pre_image_0, gt_image_0, maps[7]):
        #     lt_flatten = lt.flatten()
        #     lp_flatten = lp.flatten()
        #     wm_flatten = wm.flatten()
        #     self.adjusted_confusion_matrix_8 += self._generate_weighted_matrix(lt_flatten, lp_flatten, wm_flatten)
        
        


    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_2 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_3 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_4 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_5 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_6 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_7 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_8 = np.zeros((self.num_class,) * 2)
        self.adjusted_confusion_matrix_9 = np.zeros((self.num_class,) * 2)


'''
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        # print(type(visual_loss_map[0][250][250]), type(pre_image[0][250][250]))
        pre_image_3 = pre_image_0
        gt_image_3 = gt_image_0
        visual_loss_map[visual_loss_map == -255] = 0
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == 1] = 2
        adjusted_pre_image = np.multiply(pre_image_3, visual_loss_map, dtype=np.int64, casting='unsafe')
        adjusted_gt_image = np.multiply(gt_image_3, visual_loss_map, dtype=np.int64, casting='unsafe')
        for lp, lt in zip(adjusted_pre_image, adjusted_gt_image):
            lt_flatten = lt.flatten()
            lp_flatten = lp.flatten()
            self.adjusted_confusion_matrix_3 += self._generate_matrix(lt_flatten, lp_flatten)


        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        # print(type(visual_loss_map[0][250][250]), type(pre_image[0][250][250]))
        pre_image_4 = pre_image_0
        gt_image_4 = gt_image_0
        visual_loss_map[visual_loss_map == -255] = 0
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == 1] = 5
        adjusted_pre_image = np.multiply(pre_image_4, visual_loss_map, dtype=np.int64, casting='unsafe')
        adjusted_gt_image = np.multiply(gt_image_4, visual_loss_map, dtype=np.int64, casting='unsafe')
        for lp, lt in zip(adjusted_pre_image, adjusted_gt_image):
            lt_flatten = lt.flatten()
            lp_flatten = lp.flatten()
            self.adjusted_confusion_matrix_4 += self._generate_matrix(lt_flatten, lp_flatten)


        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        # print(type(visual_loss_map[0][250][250]), type(pre_image[0][250][250]))
        pre_image_5 = pre_image_0
        gt_image_5 = gt_image_0
        visual_loss_map[visual_loss_map == -255] = 0
        visual_loss_map[visual_loss_map == 0] = 1
        visual_loss_map[visual_loss_map == 1] = 10
        adjusted_pre_image = np.multiply(pre_image_5, visual_loss_map, dtype=np.int64, casting='unsafe')
        adjusted_gt_image = np.multiply(gt_image_5, visual_loss_map, dtype=np.int64, casting='unsafe')
        for lp, lt in zip(adjusted_pre_image, adjusted_gt_image):
            lt_flatten = lt.flatten()
            lp_flatten = lp.flatten()
            self.adjusted_confusion_matrix_5 += self._generate_matrix(lt_flatten, lp_flatten)

        
        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        # print(type(visual_loss_map[0][250][250]), type(pre_image[0][250][250]))
        pre_image_6 = pre_image_0
        gt_image_6 = gt_image_0
        pre_image_6[pre_image_6 == 0] = -1
        gt_image_6[gt_image_6 == 0] = -1
        visual_loss_map[visual_loss_map == -255] = 0
        visual_loss_map[visual_loss_map == 1] = 2
        adjusted_pre_image = np.multiply(pre_image_6, visual_loss_map, dtype=np.int64, casting='unsafe')
        adjusted_gt_image = np.multiply(gt_image_6, visual_loss_map, dtype=np.int64, casting='unsafe')
        for lp, lt in zip(adjusted_pre_image, adjusted_gt_image):
            lt_flatten = lt.flatten()
            lt_flatten = lt_flatten[(lt_flatten > 0) | (lt_flatten == -2)]
            lt_flatten[lt_flatten == -1] = 0

            lp_flatten = lp.flatten()
            lp_flatten = lp_flatten[(lp_flatten > 0) | (lp_flatten == -2)]
            lp_flatten[lp_flatten == -1] = 0
            #print(lt_flatten.shape, lp_flatten.shape)
            self.adjusted_confusion_matrix_6 += self._generate_matrix(lt_flatten, lp_flatten)


        visual_loss_map = visual_loss_map_0.reshape(1, visual_loss_map_0.shape[0], visual_loss_map_0.shape[1]).astype(np.int64)
        # print(type(visual_loss_map[0][250][250]), type(pre_image[0][250][250]))
        pre_image_7 = pre_image_0
        gt_image_7 = gt_image_0
        pre_image_7[pre_image_7 == 0] = -1
        gt_image_7[gt_image_7 == 0] = -1
        visual_loss_map[visual_loss_map == -255] = 0
        visual_loss_map[visual_loss_map == 1] = 5
        adjusted_pre_image = np.multiply(pre_image_7, visual_loss_map, dtype=np.int64, casting='unsafe')
        adjusted_gt_image = np.multiply(gt_image_7, visual_loss_map, dtype=np.int64, casting='unsafe')
        for lp, lt in zip(adjusted_pre_image, adjusted_gt_image):
            lt_flatten = lt.flatten()
            lt_flatten = lt_flatten[(lt_flatten > 0) | (lt_flatten == -5)]
            lt_flatten[lt_flatten == -1] = 0

            lp_flatten = lp.flatten()
            lp_flatten = lp_flatten[(lp_flatten > 0) | (lp_flatten == -5)]
            lp_flatten[lp_flatten == -1] = 0
            #print(lt_flatten.shape, lp_flatten.shape)
            self.adjusted_confusion_matrix_7 += self._generate_matrix(lt_flatten, lp_flatten)
'''