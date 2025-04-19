import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import json

class LiquorRecommendationEngine:
    def __init__(self, user_collection_path, bottle_dataset_path):
        # Load datasets
        self.user_collection = pd.read_csv(user_collection_path)
        self.bottle_dataset = pd.read_csv(bottle_dataset_path)
        
        # Preprocess data
        self._preprocess_data()
        
        # Extract user features
        self.user_features = self._extract_user_features()
        
        # Compute bottle similarities
        self.bottle_similarity_matrix = self._compute_bottle_similarities()
    
    def _preprocess_data(self):
        """Preprocess and clean datasets"""
        # Map CSV columns to expected structure
        self.user_collection['product.id'] = self.user_collection['product_id']
        self.user_collection['product.name'] = self.user_collection['product_name']
        self.user_collection['product.spirit'] = self.user_collection['spirit']
        self.user_collection['product.proof'] = self.user_collection['proof']
        self.user_collection['product.brand'] = self.user_collection['brand']
        self.user_collection['product.average_msrp'] = self.user_collection['average_msrp']
        self.user_collection['product.brand_id'] = self.user_collection['brand_id']

        # Convert string features to appropriate types in bottle dataset
        self.bottle_dataset['proof'] = pd.to_numeric(self.bottle_dataset['proof'], errors='coerce')
        self.bottle_dataset['avg_msrp'] = pd.to_numeric(self.bottle_dataset['avg_msrp'], errors='coerce')
        
        # Fill missing values
        self.bottle_dataset['proof'].fillna(self.bottle_dataset['proof'].median(), inplace=True)
        self.bottle_dataset['avg_msrp'].fillna(self.bottle_dataset['avg_msrp'].median(), inplace=True)
        
        # Get unique values for one-hot encoding
        self.all_spirit_types = set(self.bottle_dataset['spirit_type'].unique()) | set(self.user_collection['product.spirit'].unique())
        self.all_brands = set(self.bottle_dataset['brand_id'].unique()) | set(self.user_collection['product.brand_id'].unique())
    
    def _extract_user_features(self):
        """Extract features for each user based on their collection"""
        user_features = {}
        unique_users = self.user_collection['user_id'].unique()
        
        for user_id in unique_users:
            # Get user's bottle collection
            user_bottles = self.user_collection[self.user_collection['user_id'] == user_id]
            
            # Skip users with empty collections
            if len(user_bottles) == 0:
                continue
                
            # Price range preferences
            price_features = {
                'avg_price': user_bottles['product.average_msrp'].mean(),
                'min_price': user_bottles['product.average_msrp'].min(),
                'max_price': user_bottles['product.average_msrp'].max(),
                'price_range': user_bottles['product.average_msrp'].max() - user_bottles['product.average_msrp'].min()
            }
            
            # Spirit type preferences
            spirit_counts = user_bottles['product.spirit'].value_counts(normalize=True).to_dict()
            
            # Proof preferences
            proof_values = user_bottles['product.proof'].dropna()
            proof_features = {}
            if len(proof_values) > 0:
                proof_features = {
                    'avg_proof': proof_values.mean(),
                    'min_proof': proof_values.min(),
                    'max_proof': proof_values.max(),
                }
            else:
                proof_features = {
                    'avg_proof': 80.0,  # Default values
                    'min_proof': 80.0,
                    'max_proof': 80.0,
                }
            
            # Brand preferences
            brand_counts = user_bottles['product.brand'].value_counts(normalize=True).to_dict()
            
            # Collection diversity score (number of unique spirits / total bottles)
            diversity_score = len(spirit_counts) / len(user_bottles)
            
            # Store user features
            user_features[user_id] = {
                **price_features,
                'spirit_counts': spirit_counts,
                'proof_features': proof_features,
                'brand_counts': brand_counts,
                'diversity_score': diversity_score,
                'owned_bottles': set(user_bottles['product.id'])
            }
        
        return user_features
    
    def _one_hot_encode(self, value, all_values):
        """Simple one-hot encoding function"""
        encoding = [0] * len(all_values)
        if value in all_values:
            idx = list(all_values).index(value)
            encoding[idx] = 1
        return encoding
    
    def _compute_bottle_similarities(self):
        """Compute similarity matrix between all bottles"""
        # Create feature vectors for bottles
        bottle_features = []
        
        # Create scalers for numeric features
        price_scaler = MinMaxScaler()
        proof_scaler = MinMaxScaler()
        
        # Fit scalers
        price_scaler.fit(self.bottle_dataset[['avg_msrp']])
        proof_scaler.fit(self.bottle_dataset[['proof']])
        
        # Create feature vectors
        for _, bottle in self.bottle_dataset.iterrows():
            # Normalize numeric features
            price_norm = price_scaler.transform([[bottle['avg_msrp']]])[0][0]
            
            # Handle potential missing proof values
            if pd.isna(bottle['proof']):
                proof_norm = 0.5  # Default midpoint
            else:
                proof_norm = proof_scaler.transform([[bottle['proof']]])[0][0]
            
            # One-hot encode categorical features
            spirit_type_onehot = self._one_hot_encode(bottle['spirit_type'], self.all_spirit_types)
            brand_onehot = self._one_hot_encode(bottle['brand_id'], self.all_brands)
            
            # Combine features with weights
            # Higher weight for spirit_type as it's a primary characteristic
            feature_vector = [price_norm * 1.0, proof_norm * 0.8] + \
                             [x * 1.5 for x in spirit_type_onehot] + \
                             [x * 0.7 for x in brand_onehot]
            
            bottle_features.append(feature_vector)
        
        # Convert to numpy array
        bottle_features = np.array(bottle_features)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(bottle_features)
        return similarity_matrix
    
    def filter_by_price_range(self, user_id, tolerance=0.3):
        """Filter bottles within a price range of the user's average price"""
        # Skip if user not found
        if user_id not in self.user_features:
            return self.bottle_dataset
            
        user_avg_price = self.user_features[user_id]['avg_price']
        price_min = max(10, user_avg_price * (1 - tolerance))  # Minimum $10
        price_max = user_avg_price * (1 + tolerance)
        
        return self.bottle_dataset[(self.bottle_dataset['avg_msrp'] >= price_min) & 
                                 (self.bottle_dataset['avg_msrp'] <= price_max)]
    
    def get_similar_profile_recommendations(self, user_id, top_n=5):
        """Get recommendations similar to user's existing bottles"""
        # Skip if user not found
        if user_id not in self.user_features:
            return []
            
        # Get user's owned bottles
        owned_bottle_ids = self.user_features[user_id]['owned_bottles']
        
        # Find bottles similar to what user already has
        similar_bottles = []
        for bottle_id in owned_bottle_ids:
            # Find the bottle in the dataset
            bottle_matches = self.bottle_dataset[self.bottle_dataset['id'] == bottle_id]
            if len(bottle_matches) == 0:
                continue
                
            bottle_idx = bottle_matches.index[0]
            
            # Find most similar bottles
            similarities = self.bottle_similarity_matrix[bottle_idx]
            similar_indices = similarities.argsort()[::-1][1:21]  # Top 20 excluding itself
            
            for idx in similar_indices:
                rec_bottle_id = self.bottle_dataset.iloc[idx]['id']
                if rec_bottle_id not in owned_bottle_ids:  # Don't recommend already owned bottles
                    similar_bottles.append({
                        'bottle_id': rec_bottle_id,
                        'similarity_score': similarities[idx],
                        'name': self.bottle_dataset.iloc[idx]['name'],
                        'spirit_type': self.bottle_dataset.iloc[idx]['spirit_type'],
                        'price': self.bottle_dataset.iloc[idx]['avg_msrp'],
                        'similar_to': bottle_id  # Track which bottle it's similar to
                    })
        
        # Remove duplicates (keep highest similarity score)
        seen_bottles = {}
        for bottle in similar_bottles:
            bid = bottle['bottle_id']
            if bid not in seen_bottles or bottle['similarity_score'] > seen_bottles[bid]['similarity_score']:
                seen_bottles[bid] = bottle
        
        # Convert to list and sort by similarity score
        unique_bottles = list(seen_bottles.values())
        sorted_bottles = sorted(unique_bottles, key=lambda x: x['similarity_score'], reverse=True)
        
        return sorted_bottles[:top_n]
    
    def get_diversity_recommendations(self, user_id, top_n=3):
        """Get recommendations to diversify user's collection"""
        # Skip if user not found
        if user_id not in self.user_features:
            return []
            
        # Get user's spirit preferences
        user_spirits = set(self.user_features[user_id]['spirit_counts'].keys())
        all_spirits = set(self.bottle_dataset['spirit_type'].unique())
        
        # Find spirit types not in user's collection
        missing_spirits = all_spirits - user_spirits
        
        # For each missing spirit, find bottles within user's price range
        user_avg_price = self.user_features[user_id]['avg_price']
        price_min = max(10, user_avg_price * 0.7)  # At least 70% of user's average price
        price_max = user_avg_price * 1.3  # At most 130% of user's average price
        
        diverse_recs = []
        owned_bottle_ids = self.user_features[user_id]['owned_bottles']
        
        for spirit in missing_spirits:
            # Find bottles of this spirit in the user's price range
            spirit_bottles = self.bottle_dataset[
                (self.bottle_dataset['spirit_type'] == spirit) & 
                (self.bottle_dataset['avg_msrp'] >= price_min) & 
                (self.bottle_dataset['avg_msrp'] <= price_max)
            ]
            
            if not spirit_bottles.empty:
                # Get the most popular bottle in this spirit category
                top_bottle = spirit_bottles.sort_values('popularity', ascending=False).iloc[0]
                
                if top_bottle['id'] not in owned_bottle_ids:  # Don't recommend already owned bottles
                    diverse_recs.append({
                        'bottle_id': top_bottle['id'],
                        'name': top_bottle['name'],
                        'spirit_type': spirit,
                        'popularity': top_bottle['popularity'],
                        'price': top_bottle['avg_msrp'],
                        'recommendation_reason': 'diversify'
                    })
        
        # Sort by popularity and return top recommendations
        sorted_recs = sorted(diverse_recs, key=lambda x: x['popularity'], reverse=True)
        return sorted_recs[:top_n]
    
    def generate_recommendations(self, user_id, similar_count=3, diverse_count=2):
        """Generate final personalized recommendations for a user"""
        # Skip if user not found
        if user_id not in self.user_features:
            return []
            
        # Get recommendations filtered by price range
        price_filtered_bottles = self.filter_by_price_range(user_id)
        
        # Get similar profile recommendations
        similar_recs = self.get_similar_profile_recommendations(user_id, top_n=similar_count)
        
        # Get diversity recommendations
        diversity_recs = self.get_diversity_recommendations(user_id, top_n=diverse_count)
        
        # Combine recommendations with weights
        all_recs = []
        
        # Add similar profile recommendations with higher weight
        for rec in similar_recs:
            all_recs.append({
                'bottle_id': rec['bottle_id'],
                'name': rec['name'],
                'spirit_type': rec['spirit_type'],
                'price': rec['price'],
                'score': 0.7 * rec['similarity_score'],
                'type': 'similar_profile',
                'reason': f"Similar to {self.user_collection[self.user_collection['product.id'] == rec['similar_to']]['product.name'].values[0]} in your collection"
            })
        
        # Add diversity recommendations with lower weight
        for i, rec in enumerate(diversity_recs):
            all_recs.append({
                'bottle_id': rec['bottle_id'],
                'name': rec['name'],
                'spirit_type': rec['spirit_type'], 
                'price': rec['price'],
                'score': 0.3 * (1 - (i/len(diversity_recs))) if len(diversity_recs) > 0 else 0.3,
                'type': 'diversity',
                'reason': f"Add {rec['spirit_type']} to diversify your collection"
            })
        
        # Sort by combined score and return top recommendations
        sorted_recs = sorted(all_recs, key=lambda x: x['score'], reverse=True)
        
        # Add ranking numbers
        for i, rec in enumerate(sorted_recs):
            rec['rank'] = i + 1
            
        return sorted_recs
    
    def recommend_for_user(self, user_id, num_recommendations=5):
        """Generate formatted recommendations for a specific user"""
        recs = self.generate_recommendations(user_id)[:num_recommendations]
        
        if not recs:
            return {"error": "No recommendations available for this user"}
        
        formatted_recs = {
            "user_id": user_id,
            "recommendations": recs,
            "user_stats": {
                "collection_size": len(self.user_features[user_id]['owned_bottles']),
                "avg_bottle_price": round(self.user_features[user_id]['avg_price'], 2),
                "spirits_in_collection": list(self.user_features[user_id]['spirit_counts'].keys()),
                "diversity_score": round(self.user_features[user_id]['diversity_score'], 2)
            }
        }
        
        return formatted_recs

# Example usage
if __name__ == "__main__":
    # Initialize the recommendation engine
    rec_engine = LiquorRecommendationEngine("data/generated_user_data.csv", "data/501_Bottle_Dataset.csv")
    
    # Generate recommendations for a specific user
    user_id = 100002 #100248  # Example user ID
    recommendations = rec_engine.recommend_for_user(user_id)
    
    print(f"Recommendations for user {user_id}:")
    for rec in recommendations["recommendations"]:
        print(f"{rec['rank']}. {rec['name']} ({rec['spirit_type']}) - ${rec['price']}")
        print(f"   Reason: {rec['reason']}")
    
    print("\nUser Stats:")
    print(f"Collection size: {recommendations['user_stats']['collection_size']} bottles")
    print(f"Average bottle price: ${recommendations['user_stats']['avg_bottle_price']}")
    print(f"Spirits in collection: {', '.join(recommendations['user_stats']['spirits_in_collection'])}")
    print(f"Diversity score: {recommendations['user_stats']['diversity_score']}")