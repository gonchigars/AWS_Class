import os                       # Import the os module for accessing environment variables.
from dotenv import load_dotenv  # Import load_dotenv to load variables from a .env file.
import boto3                    # Import boto3 to interact with AWS services.
from botocore.exceptions import ClientError  # Import ClientError for catching errors from boto3 calls.

# Load environment variables from the .env file in the project root.
load_dotenv()

# Retrieve the AWS access key from the environment variables.
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
# Retrieve the AWS secret key from the environment variables.
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
# Retrieve the AWS region from the environment variables.
AWS_REGION = os.getenv("AWS_REGION")

def create_bucket(bucket_name):
    """
    Creates an S3 bucket, with special handling for the us-east-1 region.
    """
    # Ensure the AWS region is set; if not, raise a ValueError.
    if not AWS_REGION:
        raise ValueError("AWS_REGION must be set in your .env file.")

    try:
        # For the us-east-1 region, AWS S3 has special requirements:
        # - Do not specify the region_name in the boto3 client.
        # - Do not use CreateBucketConfiguration.
        if AWS_REGION == "us-east-1":
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,           # Provide the AWS access key.
                aws_secret_access_key=AWS_SECRET_KEY           # Provide the AWS secret key.
            )
            # Create the bucket without a LocationConstraint.
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            # For regions other than us-east-1, include the region name and configuration.
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,           # Provide the AWS access key.
                aws_secret_access_key=AWS_SECRET_KEY,         # Provide the AWS secret key.
                region_name=AWS_REGION                        # Specify the desired AWS region.
            )
            # Create the bucket with a CreateBucketConfiguration that includes the region.
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
            )

        # Inform the user that the bucket was successfully created.
        print(f"Bucket '{bucket_name}' created successfully in region '{AWS_REGION}'!")
        return True

    except ClientError as e:
        # If a boto3 ClientError occurs, print the error message.
        print(f"ClientError creating bucket: {e}")
        return False
    except ValueError as e:
        # Catch any ValueErrors (such as AWS_REGION not being set) and print the message.
        print(f"Configuration error: {e}")
        return False

# Only execute the following code if this script is run directly (not imported).
if __name__ == "__main__":
    # Prompt the user for the bucket name and remove any surrounding whitespace.
    bucket_name = input("Enter the bucket name: ").strip()

    # Check if the user provided a non-empty bucket name.
    if not bucket_name:
        print("Bucket name cannot be empty.")
    else:
        # Attempt to create the bucket with the provided name.
        create_bucket(bucket_name)
