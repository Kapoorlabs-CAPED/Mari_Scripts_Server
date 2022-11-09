#!/bin/bash
#!/bin/bash

aws s3 cp  s3://marioneat/Mari_Models/oneat_xenopus_resnet/ /gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/oneat_xenopus_resnet/ --recursive
aws s3 cp s3://marioneat/Mari_Models/oneat_xenopus_densenet/ /gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/oneat_xenopus_densenet/  --recursive
aws s3 cp s3://marioneat/Mari_Models/oneat_xenopus_sparsenet/ /gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/oneat_xenopus_sparsenet/  --recursive
