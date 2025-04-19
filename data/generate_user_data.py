import csv
import random
import datetime
import uuid
import string
from pathlib import Path

def generate_username():
    """Generate random usernames."""
    first_names = ["john", "sarah", "mike", "emma", "david", "lily", "alex", "olivia", 
                  "james", "sophia", "robert", "anna", "daniel", "julia", "chris", 
                  "victoria", "jake", "kate", "tom", "rachel"]
    last_names = ["smith", "jones", "wilson", "brown", "taylor", "davis", "white", 
                 "miller", "clark", "hall", "lee", "allen", "young", "king", "wright",
                 "scott", "green", "baker", "adams", "lewis"]
    
    # Sometimes use first+last, sometimes first+numbers, sometimes just first
    name_style = random.randint(1, 3)
    if name_style == 1:
        # first+last
        return random.choice(first_names) + random.choice(last_names)
    elif name_style == 2:
        # first+numbers
        return random.choice(first_names) + str(random.randint(1, 999))
    else:
        # sometimes add special characters 
        name = random.choice(first_names)
        if random.random() < 0.3:  # 30% chance to add special chars
            name += random.choice(["_", ".", "-"]) + random.choice(first_names)
        return name

def load_products(filename):
    """Load products from the bottle dataset."""
    products = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert empty strings to None for certain fields
            for key in row:
                if row[key] == '':
                    row[key] = None
            products.append(row)
    return products

def generate_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + datetime.timedelta(days=random_number_of_days)

def main():
    # Define parameters
    num_users = 1000  # Number of unique users to generate
    min_bars_per_user = 1
    max_bars_per_user = 15
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 4, 18)
    output_file = "generated_user_data.csv"
    
    # Load products from the 501 Bottle Dataset
    products = load_products("501_Bottle_Dataset.csv")
    
    # Initialize user IDs and usernames
    user_ids = [100000 + i for i in range(num_users)]
    usernames = []
    while len(usernames) < num_users:
        username = generate_username()
        if username not in usernames:
            usernames.append(username)
    
    # Prepare output file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["id", "bar_id", "user_id", "user_name", "added", "fill_percentage", 
                      "product_id", "product_name", "brand", "brand_id", "spirit", 
                      "size", "proof", "average_msrp", "fair_price", "shelf_price", "popularity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        bar_id_counter = 1000000
        
        # Generate data for each user
        for i in range(num_users):
            user_id = user_ids[i]
            username = usernames[i]
            
            # Determine how many bars this user has
            num_bars = random.randint(min_bars_per_user, max_bars_per_user)
            
            # Generate bars for this user
            for _ in range(num_bars):
                bar_id_counter += 1
                
                # Select a random product
                product = random.choice(products)
                
                # Generate bar data
                bar_data = {
                    "id": bar_id_counter,
                    "bar_id": bar_id_counter,
                    "user_id": user_id,
                    "user_name": username,
                    "added": generate_date(start_date, end_date).strftime("%Y-%m-%d"),
                    "fill_percentage": random.choice([0, 5, 10, 25, 50, 75, 100]),
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "brand": product["name"].split(" ")[0],  # Using first word of product name as brand
                    "brand_id": product["brand_id"],
                    "spirit": product["spirit_type"],
                    "size": product["size"],
                    "proof": product["proof"] if product["proof"] else (float(product["abv"]) * 2 if product["abv"] else ""),
                    "average_msrp": product["avg_msrp"],
                    "fair_price": product["fair_price"],
                    "shelf_price": product["shelf_price"],
                    "popularity": product["popularity"]
                }
                
                writer.writerow(bar_data)
    
    print(f"Generated {output_file} with data for {num_users} users.")

if __name__ == "__main__":
    main()