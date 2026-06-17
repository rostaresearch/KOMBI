#this is a bash script to run the NaCl 20 times.

#we use conda to activate the environment
conda activate biophys_env

#run the python.
for i in {1..20}
do
    nohup python explore_bias_NaCl_gen.py &
done