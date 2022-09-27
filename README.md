# transfer_ws100_code
———code_clean_transfer

This code is used to apply the climate model output (ws10,tas,tas850) on the trained RF model.The output is the ws100 and load factor, which has the same shape with the climate model.
The details of the code are as follows.

1 import packages
2 Assign a value to a variable
3 load the important dataset
4 start the application, cycle by the year
(1) prepare the prediction feature
(2) normalization
(3) apply the trained RF model
(4) Processing data format, which has the same shape as the climate model
(5）save the dataset, ws100 and load factor

Tips 1. In this example, the frequency for w10 is 3-hourly, for tas is 3-hourly, for tas850 is 6-hourly. The frequency of ws100 and load factor we saved is 6-hourly.
Thus, the uniform time frequency is very important.

Tips 2. For the w10 feature, we use w10(t-4), w10(t-1), w10(t), w10(t+1), w10(t+2), w10(t+3), w10(t+4). We need to confirm, cause the frequency of w10 is 3-hourly, thus, the w10( t-4) means w10( t- 4*3)=w10(t-12h). We need to confirm that when we use the 1-hourly dataset.

Tip3. For the time feature. The 0-23 hour, the time feature is 
array([0.        , 0.33333333, 0.66666667, 1.        , 1.33333333,
       1.66666667, 2.        , 2.33333333, 2.66666667, 3.        ,
       3.33333333, 3.66666667, 4.        , 4.33333333, 4.66666667,
       5.        , 5.33333333, 5.66666667, 6.        , 6.33333333,
       6.66666667, 7.        , 7.33333333, 7.66666667])


———function.
This code is used to calculate the load factor based on ws100 and save the dataset as the .nc format.

1) function.load_factor
calculate the load factor based on ws100

2) function.write_to_nc
save the dataset as the .nc format.

