DrivingStereo Dataset
=====================

Benchmarks:

UNet:  {'d1_all': tensor(2.5330), 'epe': tensor(0.9596), 'thres_1': tensor(27.3160), 'thres_2': tensor(5.6710), 'thres_3': tensor(2.6982)}

USAM-Net: {'d1_all': tensor(1.9442), 'epe': tensor(0.8708), 'thres_1': tensor(26.2553), 'thres_2': tensor(4.7206), 'thres_3': tensor(2.0876)}

SEG-USAM-Net{'d1_all': tensor(2.2688), 'epe': tensor(0.8819), 'thres_1': tensor(26.2567), 'thres_2': tensor(5.3471), 'thres_3': tensor(2.4171)}

SEG-UNet: {'d1_all': tensor(3.1876), 'epe': tensor(1.0458), 'thres_1': tensor(30.6258), 'thres_2': tensor(7.2369), 'thres_3': tensor(3.3572)}


KITTI2015 Dataset
=================

Before Fine-Tuning Benchmarks:

UNet: {'d1_all': tensor(8.5754), 'epe': tensor(1.6220), 'thres_1': tensor(46.0795), 'thres_2': tensor(16.3581), 'thres_3': tensor(8.8142)}

USAM-Net: {'d1_all': tensor(7.0334), 'epe': tensor(1.3059), 'thres_1': tensor(34.9043), 'thres_2': tensor(13.1788), 'thres_3': tensor(7.1896)}

Seg-Unet: {'d1_all': tensor(8.2774), 'epe': tensor(1.4695), 'thres_1': tensor(38.9731), 'thres_2': tensor(15.4233), 'thres_3': tensor(8.5295)}

SEG-USAM-Net: {'d1_all': tensor(10.0444), 'epe': tensor(1.6333), 'thres_1': tensor(43.4575), 'thres_2': tensor(17.7397), 'thres_3': tensor(10.2339)}

After Fine-Tuning Benchmarks:

UNet: { 'd1_all': tensor(5.7292), 'epe': tensor(1.2127), 'thres_1': tensor(30.3293), 'thres_2': tensor(10.7396), 'thres_3': tensor(5.9458)}

USAM-Net: {'d1_all': tensor(5.6020), 'epe': tensor(1.1103), 'thres_1': tensor(30.8308), 'thres_2': tensor(10.8129), 'thres_3': tensor(5.7980)}

SEG-Unet {'d1_all': tensor(6.1091), 'epe': tensor(1.2702), 'thres_1': tensor(32.0892), 'thres_2': tensor(11.8085), 'thres_3': tensor(6.3561)}

SEG-USAM-Net {'d1_all': tensor(6.0054), 'epe': tensor(1.2152), 'thres_1': tensor(33.9578), 'thres_2': tensor(11.7951), 'thres_3': tensor(6.1638)}