# Camouflage Detector
Term project that detects artificial camouflage patterns in natural settings and compares different machine learning approaches.
The project also evaluates how differences in the features of camouflage patterns impact the ability for models to discriminate them with natural scenery.

CADPAT (Canadian Pattern) and MARPAT (Marine Corps Pattern) easiest to detect with a small number of false positives:

<img src="https://github.com/skapura/camouflage-detector/blob/master/cadpat_marpat_output.jpg" width=400 height=300>

Pencott was the most difficult pattern to differentiate.  Although we can find all the regions containing the pattern, the features are too similar to the natural setting to do so effectively without more false positives:

<img src="https://github.com/skapura/camouflage-detector/blob/master/pencott_train_output.jpg" width=400 height=300>
