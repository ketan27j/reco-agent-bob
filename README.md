# Reco Agent Bob - Liquor Recommendation System

An intelligent recommendation system that suggests liquor bottles based on user preferences and collection patterns. The system uses collaborative filtering and content-based approaches to provide personalized recommendations for spirits and liquors.

## Features

- **Personalized Recommendations**: Generates recommendations based on:
  - Similar bottles to user's existing collection
  - Collection diversity opportunities
  - Price range preferences
  - Spirit type preferences

- **Smart Filtering**:
  - Price range filtering with adjustable tolerance
  - Exclusion of already owned bottles
  - Consideration of proof preferences
  - Brand affinity analysis

- **User Collection Analysis**:
  - Collection diversity scoring
  - Price range analysis
  - Spirit type distribution
  - Collection statistics

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ketan27j/reco-agent-bob.git
cd reco-agent-bob
```

2. Create a virtual environment using `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage - Example 1

```python
from reco_agent_bob import LiquorRecommendationEngine

# Initialize the engine
rec_engine = LiquorRecommendationEngine(
    "data/user_data.csv",
    "data/501_Bottle_Dataset.csv"
)

# Get recommendations for a user
user_id = 100248
recommendations = rec_engine.recommend_for_user(user_id)

# Print recommendations
print(f"Recommendations for user {user_id}:")
for rec in recommendations["recommendations"]:
    print(f"{rec['rank']}. {rec['name']} ({rec['spirit_type']}) - ${rec['price']}")
    print(f"   Reason: {rec['reason']}")
```
### Output

```python
Recommendations for user 100248:
1. Elijah Craig Toasted Barrel (Bourbon) - $49.72
   Reason: Similar to Heaven Hill Bottled In Bond 7 Year in your collection
2. Heaven Hill Old Style Bourbon 6 Year (Bourbon) - $12.99
   Reason: Similar to Heaven Hill Bottled In Bond 7 Year in your collection
3. Heaven Hill: Grain to Glass - Wheated Bourbon (Bourbon) - $99.0
   Reason: Similar to Heaven Hill Bottled In Bond 7 Year in your collection
4. E.H. Taylor, Jr. Straight Rye (Rye) - $79.99
   Reason: Add Rye to diversify your collection
5. Jack Daniel's Single Barrel Barrel Proof (Whiskey) - $64.18
   Reason: Add Whiskey to diversify your collection

User Stats:
Collection size: 5 bottles
Average bottle price: $67.96
Spirits in collection: Canadian Whisky, Gin, Bourbon
Diversity score: 0.6
```
### Basic Usage - Example 2

generated_user_data.csv file is generated using data/generate_user_data.py for testing purpose.

```python
from reco_agent_bob import LiquorRecommendationEngine

# Initialize the engine
rec_engine = LiquorRecommendationEngine(
    "data/generated_user_data.csv",
    "data/501_Bottle_Dataset.csv"
)

# Get recommendations for a user
user_id = 100004
recommendations = rec_engine.recommend_for_user(user_id)

# Print recommendations
print(f"Recommendations for user {user_id}:")
for rec in recommendations["recommendations"]:
    print(f"{rec['rank']}. {rec['name']} ({rec['spirit_type']}) - ${rec['price']}")
    print(f"   Reason: {rec['reason']}")
```

### Output

```python
Recommendations for user 100004:
1. Larceny Barrel Proof Batch B522 (Bourbon) - $59.99
   Reason: Similar to Larceny Barrel Proof Batch C923 in your collection
2. Larceny Barrel Proof Batch B523 (Bourbon) - $59.0
   Reason: Similar to Larceny Barrel Proof Batch C923 in your collection
3. Jack Daniel's Bottled in Bond (Whiskey) - $37.99
   Reason: Similar to Jack Daniel’s Triple Mash in your collection
4. E.H. Taylor, Jr. Straight Rye (Rye) - $79.99
   Reason: Add Rye to diversify your collection
5. Nikka Coffey Grain Japanese Whisky (Japanese Whisky) - $70.92
   Reason: Add Japanese Whisky to diversify your collection

User Stats:
Collection size: 10 bottles
Average bottle price: $97.13
Spirits in collection: Bourbon, Whiskey
Diversity score: 0.2
```
### Data Format

#### User Collection Data (CSV)
Required columns:
- user_id: Unique identifier for each user
- product_id: Unique identifier for each product
- product_name: Name of the product
- brand: Brand name
- brand_id: Unique identifier for each brand
- spirit: Type of spirit
- proof: Alcohol proof
- average_msrp: Average retail price

#### Bottle Dataset (CSV)
Required columns:
- id: Unique identifier for each bottle
- name: Bottle name
- spirit_type: Category of spirit
- brand_id: Brand identifier
- proof: Alcohol proof
- avg_msrp: Average retail price
- popularity: Popularity score

## How It Works

The recommendation engine uses a hybrid approach combining:

1. **Content-Based Filtering**:
   - Analyzes bottle characteristics (spirit type, proof, price)
   - Computes similarity between bottles using cosine similarity
   - Considers user's existing collection patterns

2. **Diversity-Based Recommendations**:
   - Identifies gaps in user's collection
   - Suggests popular bottles from unexplored spirit categories
   - Maintains price range consistency

3. **Price Range Analysis**:
   - Adapts recommendations to user's spending patterns
   - Applies flexible price tolerance
   - Ensures suggestions remain within reasonable price bounds

## Output Format

The recommendation output includes:

1. Personalized bottle recommendations with:
   - Ranking
   - Bottle name and spirit type
   - Price
   - Recommendation reason

2. User statistics:
   - Collection size
   - Average bottle price
   - Spirit types in collection
   - Collection diversity score

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
