**Respiratory Rate Detection**

This approach for Respiratory rate detection uses optical flow to find motion of the chest and body region. It then subtracts the body movements from the chest movemenets to account for test subject movement.
Various filters are used to smoothen the curve and remove the noise and finally the number of peaks are found and Respiratory Rate is calculated using the formula:

Repiratory rate= $(NumOfPeaks/Duration(in seconds))*60$
