from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor


def get_transforms(name="nnunet_default"):
    transforms = []

    if name == "nnunet_default":

        transforms += [SpatialTransform(
            None,
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0.0, 200.0),
            sigma=(9.0, 13.0),
            do_rotation=True, angle_x=(-0.2617993877991494, 0.2617993877991494), angle_y=(-0.0, 0.0),
            angle_z=(-0.0, 0.0), p_rot_per_axis=1,
            do_scale=True, scale=(0.9, 1.1),
            border_mode_data='constant', border_cval_data=0, order_data=3,
            border_mode_seg="constant", border_cval_seg=0,
            order_seg=0, random_crop=False, p_el_per_sample=0.2,
            p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=True
        )]
        transforms += [
            GaussianNoiseTransform(p_per_sample=0.1),
            GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                  p_per_channel=0.5),
            BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)
        ]
        transforms += [(ContrastAugmentationTransform(p_per_sample=0.15)),
                       SimulateLowResolutionTransform(zoom_range=(0.5, 1),
                                                      per_channel=True, p_per_channel=0.5,
                                                      order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                      ignore_axes=None)]
        transforms += [
            GammaTransform((0.7, 1.5),
                           True,
                           True,
                           retain_stats=True,
                           p_per_sample=0.1)
        ]

        transforms += [
            GammaTransform((0.7, 1.5),
                           False,
                           True,
                           retain_stats=True,
                           p_per_sample=0.3)
        ]

        transforms += [
            MirrorTransform((0, 1))
        ]

        transforms += [
            NumpyToTensor(cast_to='float')
        ]

    elif name == "nnunet_brats":

        transforms += [SpatialTransform(
            None,
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0.0, 200.0),
            sigma=(9.0, 13.0),
            do_rotation=True, angle_x=(-0.2617993877991494, 0.2617993877991494), angle_y=(-0.0, 0.0),
            angle_z=(-0.0, 0.0), p_rot_per_axis=1,
            do_scale=True, scale=(0.65, 1.6),
            border_mode_data='constant', border_cval_data=0, order_data=3,
            border_mode_seg="constant", border_cval_seg=0,
            order_seg=0, random_crop=False, p_el_per_sample=0.3,
            p_scale_per_sample=0.3, p_rot_per_sample=0.3,
            independent_scale_for_each_axis=True
        )]
        transforms += [
            GaussianNoiseTransform(p_per_sample=0.1),
            GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                  p_per_channel=0.5),
            BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)
        ]
        transforms += [(ContrastAugmentationTransform(p_per_sample=0.15)),
                       SimulateLowResolutionTransform(zoom_range=(0.5, 1),
                                                      per_channel=True, p_per_channel=0.5,
                                                      order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                      ignore_axes=None)]
        transforms += [
            GammaTransform((0.7, 1.5),
                           True,
                           True,
                           retain_stats=True,
                           p_per_sample=0.1)
        ]

        transforms += [
            GammaTransform((0.7, 1.5),
                           False,
                           True,
                           retain_stats=True,
                           p_per_sample=0.3)
        ]

        transforms += [

        ]

        transforms += [
            MirrorTransform((0, 1))
        ]

        transforms += [
            NumpyToTensor(cast_to='float')
        ]

    elif name == "cyclegan":
        transforms += [SpatialTransform(
            None,
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0.0, 200.0),
            sigma=(9.0, 13.0),
            do_rotation=True, angle_x=(-0.2617993877991494, 0.2617993877991494), angle_y=(-0.0, 0.0),
            angle_z=(-0.0, 0.0), p_rot_per_axis=1,
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0, order_data=3,
            border_mode_seg="constant", border_cval_seg=0,
            order_seg=0, random_crop=False, p_el_per_sample=0.3,
            p_scale_per_sample=0.3, p_rot_per_sample=0.3,
            independent_scale_for_each_axis=True
        )]
        transforms += [
            MirrorTransform((0, 1))
        ]
        transforms += [
            NumpyToTensor(cast_to='float')
        ]

    elif name == "validation":
        transforms += [
            NumpyToTensor(cast_to='float')
        ]

    else:
        raise NotImplementedError()

    return Compose(transforms)
