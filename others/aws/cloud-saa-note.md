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

------------------------------------------------------------------------------------------------------------------------------

2. A company runs an online voting system for a weekly live television program. During broadcasts, 
users submit hundreds of thousands of votes within minutes to a front-end fleet of Amazon EC2 
instances that run in an Auto Scaling group. The EC2 instances write the votes to an Amazon RDS 
database. However, the database is unable to keep up with the requests that come from the EC2 
instances. A solutions architect must design a solution that processes the votes in the most efficient 
manner and without downtime. 
Which solution meets these requirements? 
- A) Migrate the front-end application to AWS Lambda. Use Amazon API Gateway to route user requests to 
the Lambda functions. 
- B) Scale the database horizontally by converting it to a Multi-AZ deployment. Configure the front-end 
application to write to both the primary and secondary DB instances. 
- C) Configure the front-end application to send votes to an Amazon Simple Queue Service (Amazon SQS) 
queue. Provision worker instances to read the SQS queue and write the vote information to the database. 
- D) Use Amazon EventBridge (Amazon CloudWatch Events) to create a scheduled event to re-provision the 
database with larger, memory optimized instances during voting periods. When voting ends, re-provision 
the database to use smaller instances. 

C – Decouple the ingestion of votes from the database to allow the voting system to continue processing votes 
without waiting for the database writes. Add dedicated workers to read from the SQS queue to allow votes to be 
entered into the database at a controllable rate. The votes will be added to the database as fast as the database 
can process them, but no votes will be lost.

------------------------------------------------------------------------------------------------------------------------------

3. A company has a two-tier application architecture that runs in public and private subnets. Amazon EC2 
instances running the web application are in the public subnet and an EC2 instance for the database runs 
on the private subnet. The web application instances and the database are running in a single Availability 
Zone (AZ). 
Which combination of steps should a solutions architect take to provide high availability for this 
architecture? (Select TWO.) 
- A) Create new public and private subnets in the same AZ. 
- B) Create an Amazon EC2 Auto Scaling group and Application Load Balancer spanning multiple AZs for the 
web application instances. 
- C) Add the existing web application instances to an Auto Scaling group behind an Application Load 
Balancer. 
- D) Create new public and private subnets in a new AZ. Create a database using an EC2 instance in the 
public subnet in the new AZ. Migrate the old database contents to the new database. 
- E) Create new public and private subnets in the same VPC, each in a new AZ. Create an Amazon RDS 
Multi-AZ DB instance in the private subnets. Migrate the old database contents to the new DB instance.

B, E – Create new subnets in a new Availability Zone (AZ) to provide a redundant network. Create an Auto 
Scaling group with instances in two AZs behind the load balancer to ensure high availability of the web application 
and redistribution of web traffic between the two public AZs. Create an RDS DB instance in the two private 
subnets to make the database tier highly available too. 

The correct answers are:  

### **Explanation:**  
To achieve **high availability (HA)**, the architecture must be resilient across **multiple Availability Zones (AZs)**. Here’s why these options are correct:  

1. **Option B (Auto Scaling + Load Balancer for Web Tier)**  
   - **Auto Scaling** ensures the web application can scale out/in based on demand.  
   - **Application Load Balancer (ALB)** distributes traffic across instances in **multiple AZs**, improving fault tolerance.  
   - This ensures the web tier remains available even if one AZ fails.  

2. **Option E (Multi-AZ RDS for Database Tier)**  
   - **Amazon RDS Multi-AZ** provides automatic failover to a standby database in a **different AZ** if the primary fails.  
   - Moving from a **single EC2-based database** (in a private subnet) to **RDS Multi-AZ** is a best practice for HA.  
   - New subnets in a **new AZ** ensure the database is distributed across AZs.  

### **Why Not the Other Options?**  
- **A) Same AZ subnets** → Does **not** improve HA (still single point of failure).  
- **C) Adding existing instances to ASG/ALB** → Doesn’t address **multi-AZ redundancy** (they’re still in one AZ).  
- **D) New database in a public subnet** → **Security risk!** Databases should **never** be in public subnets.  

### **Summary:**  
- **Web Tier:** Auto Scaling + ALB across AZs (B).  
- **Database Tier:** Migrate to **RDS Multi-AZ** in private subnets (E).  

------------------------------------------------------------------------------------------------------------------------------