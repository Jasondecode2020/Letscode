1. Amazon EMR (Elastic MapReduce)
Purpose: A big data platform for processing and analyzing massive datasets using open-source frameworks like Apache Spark, Hadoop, Hive, and Presto.

Key Features:

Scalable & Managed: Automatically provisions and scales clusters (EC2 instances) for distributed processing.

Supports Multiple Frameworks: Spark, Hadoop, Flink, HBase, etc.

Cost-Effective: Offers Spot Instances and auto-termination to reduce costs.

Integration: Works with S3, DynamoDB, and AWS Glue for data storage/ETL.

Use Cases:

Log analysis, machine learning, genomic research, financial modeling.

2. Amazon EBS (Elastic Block Store)
Purpose: Provides persistent block storage volumes for EC2 instances (like virtual hard drives).

Key Features:

High Performance: SSD-backed (gp3, io1) or HDD-backed (st1, sc1) options.

Durability: Replicated within an Availability Zone (AZ).

Snapshots: Back up data to S3 (cross-AZ/cross-region).

Elastic Volumes: Resize/modify volumes without downtime.

Use Cases:

Boot volumes for EC2, databases (RDS, self-managed), enterprise apps.

3. Snowmobile

Purpose: Designed for exabyte-scale (EB) or massive petabyte-scale data transfers (like 60PB).

How it works: A 45-foot ruggedized shipping container is physically delivered to your data center, loaded with data, and returned to AWS for upload.

Speed: Can transfer up to 100PB per Snowmobile (much faster than internet-based transfers).

Use case: Ideal for migrations of huge datasets (e.g., media libraries, genomic data, backups).

4. Amazon EBS

Low-Latency Performance: Designed for frequent read/write operations (ideal for databases like MySQL, PostgreSQL, Oracle, or RDS).

High IOPS & Throughput:

gp3 (SSD): Default for balanced performance (up to 16,000 IOPS and 1,000 MB/s throughput per volume).

io1/io2 (Provisioned IOPS SSD): Supports up to 256,000 IOPS for high-transaction databases.

Durability: Replicated within an Availability Zone (AZ).

Integration: Native support for EC2 and RDS (AWS-managed databases).

5. Amazon Aurora

✅ 5x Performance Over MySQL:

Aurora is AWS’s high-performance, MySQL-compatible database engine, optimized for low-latency queries and high throughput.

Uses a distributed, cloud-native architecture (separates compute and storage) to avoid bottlenecks of traditional MySQL.

Delivers up to 5x higher throughput than standard MySQL (AWS benchmark claims).

✅ Key Features:

Auto-scaling storage (up to 128TB per instance).

High availability with multi-AZ replication.

Low-cost replicas (up to 15 read replicas vs. MySQL’s 5).

6. What AWS Service Catalog Provides:

✅ Centralized IT Service Management:

Allows organizations to create and manage approved catalogs of AWS resources (e.g., EC2 instances, S3 buckets, RDS databases) for users to deploy.

Ensures compliance with company policies, security standards, and cost controls.

✅ Key Features:

Pre-defined templates (using AWS CloudFormation).

Role-based access control (RBAC) to restrict what users can deploy.

Version control for updates to approved services.

7. What is AWS CAF?

The AWS Cloud Adoption Framework (CAF) is a structured guide developed by AWS Professional Services to help organizations plan and accelerate their cloud adoption journey. It provides best practices, documentation, and tools across six key perspectives:

Business (Strategy, ROI, KPIs)

People (Training, organizational change)

Governance (Compliance, risk management)

Platform (Architecture, AWS services)

Security (Identity, encryption, threat detection)

Operations (Monitoring, incident response)

8. AWS TCO (Total Cost of Ownership) Calculator

Why AWS TCO Calculator?
Purpose: Compares the total cost of running workloads on-premises vs. AWS over 3–5 years, factoring in:

Hardware/software costs (servers, storage, networking).

Data center expenses (power, cooling, physical security).

Labor (IT staff, maintenance).

AWS pricing (EC2, S3, etc.).

Output: Generates a detailed cost savings report to justify migration.

1. 90%

4. read carefully
8. determined
32. hard 
44. two
47. two

2. 94%

16. don't knows
33. 
46. should not

3. 90%

4. 

5. 

14. 

22. too hard 

36. don't know

4. 94%

1.

19. catalog
35. 



