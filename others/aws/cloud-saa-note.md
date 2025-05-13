1. A company uses Amazon EC2 Reserved Instances to run its data processing workload. The nightly job 
typically takes 7 hours to run and must finish within a 10-hour time window. The company anticipates 
temporary increases in demand at the end of each month that will cause the job to run over the time limit 
with the capacity of the current resources. Once started, the processing job cannot be interrupted before 
completion. The company wants to implement a solution that would provide increased resource capacity 
as cost-effectively as possible. 

What should a solutions architect do to accomplish this? 
- A) Deploy On-Demand Instances during periods of high demand. 
- B) Create a second EC2 reservation for additional instances. 
- C) Deploy Spot Instances during periods of high demand. 
- D) Increase the EC2 instance size in the EC2 reservation to support the increased workload.

A – While Spot Instances would be the least costly option, they are not suitable for jobs that cannot be 
interrupted or must complete within a certain time period. On-Demand Instances would be billed for the number of 
seconds they are running. 