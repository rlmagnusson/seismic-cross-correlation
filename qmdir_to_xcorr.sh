#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --mail-user=<rm@isor.is> #
#SBATCH --partition=64cpu_256mem # request node from a specific partition
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=64 # 64 cores per node
#SBATCH --mem-per-cpu=3900 # MB RAM per cpu core
#SBATCH --time=4-00:00:00 # run for 4 days maximum
#SBATCH --output=output_xcorr.log
#SBATCH --error=errors_xcorr.log # Logs if job crashes

# qm_dir=/users/home/Rognvaldur.Magnusson/rmQM/lhr2023/QM_run
qm_dir=/users/home/Rognvaldur.Magnusson/main/xcorr/240920_tobba_xcorr
run_name=Run1_no_CAM_stations
# run_name=tiny
sac_script=/users/home/Rognvaldur.Magnusson/rmQM/sort_input_QM_parallel.py
add_id_script=/users/home/Rognvaldur.Magnusson/rmQM/add_ID.py
xcorr_script=/hpcisor/data/xcorr/xcorr.py
n_proc=60
# sac_dir_name=/users/home/Rognvaldur.Magnusson/main/xcorr/240920_tobba_xcorr/SAC


# Set up /scratch/
# Location of scratch directory on the compute nodes
scratchlocation=/scratch/users
# Create a user directory if it does not exist
if [ ! -d $scratchlocation/$USER ]; then
mkdir -p $scratchlocation/$USER
fi
# Create a temporary directory with a unique identifier associated with your jobid
tdir=$(mktemp -d $scratchlocation/$USER/$SLURM_JOB_ID-XXXX)
# make diectories in temporary directory on node

# Go to the temporary directory
cd $tdir
# Exit if tdir does not exist
if [ ! -d $tdir ]; then
echo "Temporary scratch directory does not exist ..."
echo "Something is wrong, contact support."
exit
fi

# script=sort_input_QM_parallel.py
cp -v $sac_script .
cp -v $add_id_script .
cp -v $xcorr_script .
date
echo "copying QM dir"
echo "cp -r $qm_dir/$run_name ."
cp -r $qm_dir/$run_name .
date
echo "done"


step1=`basename $add_id_script`
step2=`basename $sac_script`
step3=`basename $xcorr_script`
# local_qm=`basename $qm_dir` 

# create event list
event_dir="$run_name/locate/events"
echo "event dir: $event_dir"
cat $run_name/locate/events/* | grep -v EventID > events.lst
wc events.lst
python $step1 events.lst # creates events.lst_wID

# DEBUG
# mv events.lst_wID bla
# head -1700 bla > events.lst_wID

# split -l 1000 -a 3 -d events.lst_wID event_batch

ls -lh
module use /hpcapps/libbio-gpu/modules/all
module load Python


echo "obspy version:"
python -c "import obspy; print(obspy.__version__)"

echo "number of events: $(wc -l events.lst_wID)"
date
echo "START CONVERSION"
python $step2 events.lst_wID "$run_name/locate" ./SAC $n_proc
echo "CONVERSION COMPLETE"
date

echo "CROSS CORRELATION"
mkdir xcorr_outputs
for station in ./SAC/STATION/*
do
    python $step3 $station
done
date
echo "COPYING..."
cp  events.lst_wID $sac_dir_name -v
cp -r xcorr_outputs $qm_dir
# echo "cp -r $sac_dir_name /users/home/Rognvaldur.Magnusson/rmQM"
# cp -r $sac_dir_name /users/home/Rognvaldur.Magnusson/rmQM
echo "done"

# If the program produces many output files you can add a separate line for each file.
# Please try to only copy the files that you need.
# IMPORTANT. Delete the temporary directory and all of its content
rm -rf $tdir

