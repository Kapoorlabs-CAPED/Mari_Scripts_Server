#!/bin/bash
export PATH="/gpfslocalsup/spack_soft/environment-modules/4.3.1/gcc-4.8.5-ism7cdy4xverxywj27jvjstqwk5oxe2v/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/c3/bin:/usr/lpp/mmfs/bin:/sbin:/bin:/gpfslocalsys/slurm/current/bin:/gpfslocalsup/bin:/gpfslocalsys/bin:/gpfslocalsys/idrzap/current/bin"
source /linkhome/rech/gengoq01/uzj81mi/.bashrc
echo $LD_LIBRARY_PATH
ls -ltr
set -x
sbatch /gpfswork/rech/jsy/uzj81mi/Mari_Scripts_Server/jobscript_oneat_train
