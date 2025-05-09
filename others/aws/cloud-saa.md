Question 1:

You have purchased mycoolcompany.com on Amazon Route 53 Registrar and would like the domain to point to your Elastic Load Balancer my-elb-1234567890.us-west-2.elb.amazonaws.com. Which Route 53 Record type must you use here?

> A. CNAME
> B. Alias

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

