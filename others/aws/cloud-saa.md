Question 1:

You have purchased mycoolcompany.com on Amazon Route 53 Registrar and would like the domain to point to your Elastic Load Balancer my-elb-1234567890.us-west-2.elb.amazonaws.com. Which Route 53 Record type must you use here?

- A. CNAME
- B. Alias

Answer:

To point your domain **mycoolcompany.com** to your Elastic Load Balancer (**my-elb-1234567890.us-west-2.elb.amazonaws.com**) in Amazon Route 53, you should use an **Alias record**.

### Why Alias over CNAME?
1. **Alias records** are a Route 53-specific feature that can point directly to AWS resources (like ELB, CloudFront, S3, etc.).
2. **Alias records can be used for the root domain (apex zone, e.g., `mycoolcompany.com`)** whereas **CNAME records cannot** (they only work for subdomains like `www.mycoolcompany.com`).
3. **Alias records are free** and have better performance since Route 53 resolves them internally.

### How to Set It Up:
- **Record Type:** `A` (IPv4) or `AAAA` (IPv6)  
- **Alias Target:** Select your ELB (**my-elb-1234567890.us-west-2.elb.amazonaws.com**) from the dropdown.

### CNAME Limitation:
- If you were setting up a subdomain (e.g., `www.mycoolcompany.com`), you *could* use a **CNAME**, but for the root domain (`mycoolcompany.com`), **Alias is the correct choice**.

**Final Answer:** Use an **Alias record**.

Question 2:

You have deployed a new Elastic Beanstalk environment and would like to direct 5% of your production traffic to this new environment. This allows you to monitor for CloudWatch metrics and ensuring that there're no bugs exist with your new environment. Which Route 53 Routing Policy allows you to do so?

Answer:

Weighted Routing Policy allows you to redirect part of the traffic based on weight (e.g., percentage). It's a common use case to send part of traffic to a new version of your application.

Question 3:

You have updated a Route 53 Record's myapp.mydomain.com value to point to a new Elastic Load Balancer, but it looks like users are still redirected to the old ELB. What is a possible cause for this behavior?

Answer:

Each DNS record has a TTL (Time To Live) which orders clients for how long to cache these values and not overload the DNS Resolver with DNS requests. The TTL value should be set to strike a balance between how long the value should be cached vs. how many requests should go to the DNS Resolver.

Question 4:

You have an application that's hosted in two different AWS Regions us-west-1 and eu-west-2. You want your users to get the best possible user experience by minimizing the response time from application servers to your users. Which Route 53 Routing Policy should you choose?

Answer:

Latency Routing Policy will evaluate the latency between your users and AWS Regions, and help them get a DNS response that will minimize their latency (e.g. response time)

Question 5:

You have a legal requirement that people in any country but France should NOT be able to access your website. Which Route 53 Routing Policy helps you in achieving this?

Answer:

Geolocation

Question 6:

You have purchased a domain on GoDaddy and would like to use Route 53 as the DNS Service Provider. What should you do to make this work?

Answer:

Question 7:

Public Hosted Zones are meant to be used for people requesting your website through the Internet. Finally, NS records must be updated on the 3rd party Registrar.

What does this CIDR 10.0.4.0/28 correspond to?

Total IP Addresses: 16 (calculated as 2^(32-28) = 16)