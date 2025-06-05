import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Machine Learning predictor for construction cost estimation
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = [
            'quantity', 'project_type_encoded', 'material_category_encoded',
            'location_encoded', 'season', 'complexity_score', 'market_factor'
        ]
        self.target_column = 'unit_cost'
        self.models_trained = {}
    
    def prepare_training_data(self, db_manager) -> pd.DataFrame:
        """
        Prepare training data from historical projects
        """
        try:
            # Get historical data from database
            historical_data = db_manager.get_historical_data()
            
            if not historical_data:
                # Create synthetic training data for demonstration
                logger.warning("No historical data found, creating synthetic training data")
                return self._create_synthetic_data()
            
            df = pd.DataFrame(historical_data)
            
            # Feature engineering
            df = self._engineer_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """
        Create synthetic training data for initial model training
        """
        np.random.seed(42)
        
        materials = ['Concrete/Masonry', 'Steel/Concrete', 'Roofing Materials', 
                    'Doors/Windows', 'Finishes', 'MEP Systems', 'Miscellaneous']
        project_types = ['Residential', 'Commercial', 'Industrial', 'Infrastructure']
        locations = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília', 'Salvador']
        
        n_samples = 1000
        data = []
        
        for _ in range(n_samples):
            material = np.random.choice(materials)
            project_type = np.random.choice(project_types)
            location = np.random.choice(locations)
            
            # Base unit costs by material (R$ per unit)
            base_costs = {
                'Concrete/Masonry': 150 + np.random.normal(0, 20),
                'Steel/Concrete': 800 + np.random.normal(0, 100),
                'Roofing Materials': 45 + np.random.normal(0, 10),
                'Doors/Windows': 500 + np.random.normal(0, 75),
                'Finishes': 80 + np.random.normal(0, 15),
                'MEP Systems': 200 + np.random.normal(0, 30),
                'Miscellaneous': 100 + np.random.normal(0, 25)
            }
            
            # Project type multipliers
            type_multipliers = {
                'Residential': 1.0,
                'Commercial': 1.2,
                'Industrial': 1.4,
                'Infrastructure': 1.6
            }
            
            # Location multipliers
            location_multipliers = {
                'São Paulo': 1.2,
                'Rio de Janeiro': 1.1,
                'Belo Horizonte': 1.0,
                'Brasília': 1.05,
                'Salvador': 0.95
            }
            
            quantity = np.random.lognormal(2, 1)  # Log-normal distribution for quantities
            base_cost = base_costs[material]
            
            # Apply multipliers and add noise
            unit_cost = (base_cost * 
                        type_multipliers[project_type] * 
                        location_multipliers[location] * 
                        (1 + np.random.normal(0, 0.1)))  # 10% noise
            
            # Ensure positive costs
            unit_cost = max(unit_cost, 10)
            
            data.append({
                'material_category': material,
                'project_type': project_type,
                'location': location,
                'quantity': quantity,
                'unit_cost': unit_cost,
                'project_date': datetime.now() - timedelta(days=np.random.randint(1, 365*3))
            })
        
        df = pd.DataFrame(data)
        return self._engineer_features(df)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for machine learning
        """
        try:
            # Encode categorical variables
            if 'project_type' in df.columns:
                if 'project_type' not in self.encoders:
                    self.encoders['project_type'] = LabelEncoder()
                    df['project_type_encoded'] = self.encoders['project_type'].fit_transform(df['project_type'])
                else:
                    df['project_type_encoded'] = self.encoders['project_type'].transform(df['project_type'])
            
            if 'material_category' in df.columns:
                if 'material_category' not in self.encoders:
                    self.encoders['material_category'] = LabelEncoder()
                    df['material_category_encoded'] = self.encoders['material_category'].fit_transform(df['material_category'])
                else:
                    df['material_category_encoded'] = self.encoders['material_category'].transform(df['material_category'])
            
            if 'location' in df.columns:
                if 'location' not in self.encoders:
                    self.encoders['location'] = LabelEncoder()
                    df['location_encoded'] = self.encoders['location'].fit_transform(df['location'])
                else:
                    df['location_encoded'] = self.encoders['location'].transform(df['location'])
            
            # Time-based features
            if 'project_date' in df.columns:
                df['project_date'] = pd.to_datetime(df['project_date'])
                df['season'] = df['project_date'].dt.month % 12 // 3  # 0-3 for seasons
            else:
                df['season'] = np.random.randint(0, 4, len(df))
            
            # Complexity score based on quantity
            df['complexity_score'] = np.log1p(df['quantity']) if 'quantity' in df.columns else 1.0
            
            # Market factor (simplified)
            df['market_factor'] = 1.0 + np.random.normal(0, 0.05, len(df))  # Market volatility
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def train_models(self, db_manager):
        """
        Train ML models for each material category
        """
        try:
            logger.info("Preparing training data...")
            df = self.prepare_training_data(db_manager)
            
            if df.empty:
                raise ValueError("No training data available")
            
            # Train model for each material category
            material_categories = df['material_category'].unique()
            
            for material in material_categories:
                logger.info(f"Training model for {material}...")
                
                # Filter data for this material
                material_data = df[df['material_category'] == material].copy()
                
                if len(material_data) < 10:  # Need minimum samples
                    logger.warning(f"Insufficient data for {material}, using global model")
                    continue
                
                # Prepare features and target
                X = material_data[self.feature_columns]
                y = material_data[self.target_column]
                
                # Handle missing values
                X = X.fillna(X.mean())
                
                # Scale features
                if material not in self.scalers:
                    self.scalers[material] = StandardScaler()
                
                X_scaled = self.scalers[material].fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train ensemble of models
                models = {
                    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gb': GradientBoostingRegressor(random_state=42),
                    'xgb': xgb.XGBRegressor(random_state=42),
                    'lr': LinearRegression()
                }
                
                best_model = None
                best_score = float('inf')
                
                for name, model in models.items():
                    try:
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test)
                        mape = mean_absolute_percentage_error(y_test, y_pred)
                        
                        if mape < best_score:
                            best_score = mape
                            best_model = model
                            
                    except Exception as e:
                        logger.warning(f"Error training {name} for {material}: {str(e)}")
                        continue
                
                if best_model is not None:
                    self.models[material] = best_model
                    self.models_trained[material] = True
                    logger.info(f"Best model for {material}: MAPE = {best_score:.3f}")
                else:
                    logger.warning(f"Failed to train model for {material}")
            
            logger.info(f"Successfully trained models for {len(self.models)} material categories")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def predict_costs(self, project_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Predict costs for a project
        """
        try:
            predictions = {}
            
            for material, quantity in project_data['materials'].items():
                if material in self.models:
                    # Prepare features
                    features = self._prepare_prediction_features(
                        material, quantity, project_data
                    )
                    
                    # Scale features
                    if material in self.scalers:
                        features_scaled = self.scalers[material].transform([features])
                    else:
                        features_scaled = [features]
                    
                    # Make prediction
                    predicted_unit_cost = self.models[material].predict(features_scaled)[0]
                    total_cost = predicted_unit_cost * quantity
                    
                    # Calculate confidence (simplified)
                    confidence = self._calculate_confidence(material, features)
                    
                    predictions[material] = {
                        'unit_cost': max(predicted_unit_cost, 1.0),  # Ensure positive
                        'total_cost': max(total_cost, quantity),     # Ensure positive
                        'quantity': quantity,
                        'confidence': confidence
                    }
                else:
                    # Use fallback prediction for untrained materials
                    predictions[material] = self._fallback_prediction(material, quantity)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {}
    
    def _prepare_prediction_features(self, material: str, quantity: float, 
                                   project_data: Dict[str, Any]) -> List[float]:
        """
        Prepare features for prediction
        """
        try:
            # Encode categorical features
            project_type_encoded = 0
            if 'project_type' in self.encoders:
                try:
                    project_type_encoded = self.encoders['project_type'].transform([project_data.get('project_type', 'Residential')])[0]
                except ValueError:
                    project_type_encoded = 0
            
            material_encoded = 0
            if 'material_category' in self.encoders:
                try:
                    material_encoded = self.encoders['material_category'].transform([material])[0]
                except ValueError:
                    material_encoded = 0
            
            location_encoded = 0
            if 'location' in self.encoders:
                try:
                    location_encoded = self.encoders['location'].transform([project_data.get('location', 'São Paulo')])[0]
                except ValueError:
                    location_encoded = 0
            
            # Calculate other features
            season = datetime.now().month % 12 // 3
            complexity_score = np.log1p(quantity)
            market_factor = 1.0  # Current market factor
            
            features = [
                quantity,
                project_type_encoded,
                material_encoded,
                location_encoded,
                season,
                complexity_score,
                market_factor
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return [0.0] * len(self.feature_columns)
    
    def _calculate_confidence(self, material: str, features: List[float]) -> float:
        """
        Calculate prediction confidence (simplified)
        """
        try:
            # This is a simplified confidence calculation
            # In practice, you might use prediction intervals or ensemble variance
            
            base_confidence = 85.0  # Base confidence percentage
            
            # Adjust based on material training data availability
            if material in self.models_trained and self.models_trained[material]:
                confidence = base_confidence + np.random.normal(5, 2)  # Trained model bonus
            else:
                confidence = base_confidence - 10  # Penalty for fallback
            
            # Ensure confidence is in valid range
            confidence = max(60.0, min(98.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {str(e)}")
            return 75.0  # Default confidence
    
    def _fallback_prediction(self, material: str, quantity: float) -> Dict[str, float]:
        """
        Fallback prediction for materials without trained models
        """
        # Default unit costs by material category (R$)
        default_costs = {
            'Concrete/Masonry': 150,
            'Steel/Concrete': 800,
            'Roofing Materials': 45,
            'Doors/Windows': 500,
            'Finishes': 80,
            'MEP Systems': 200,
            'Miscellaneous': 100
        }
        
        unit_cost = default_costs.get(material, 100)
        total_cost = unit_cost * quantity
        
        return {
            'unit_cost': unit_cost,
            'total_cost': total_cost,
            'quantity': quantity,
            'confidence': 60.0  # Lower confidence for fallback
        }
    
    def update_predictions(self, project_id: int):
        """
        Update predictions for a specific project
        """
        try:
            from database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Get project data
            project_data = db_manager.get_project_for_prediction(project_id)
            
            if not project_data:
                raise ValueError(f"Project {project_id} not found")
            
            # Make predictions
            predictions = self.predict_costs(project_data)
            
            # Store predictions in database
            for material, prediction in predictions.items():
                db_manager.store_prediction(
                    project_id, 
                    material, 
                    prediction['unit_cost'],
                    prediction['total_cost'],
                    prediction['confidence']
                )
            
            logger.info(f"Updated predictions for project {project_id}")
            
        except Exception as e:
            logger.error(f"Error updating predictions: {str(e)}")
            raise
    
    def get_model_accuracy(self, project_id: int) -> Dict[str, float]:
        """
        Get model accuracy metrics for a project
        """
        try:
            from database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Get actual vs predicted costs
            accuracy_data = db_manager.get_accuracy_data(project_id)
            
            accuracies = {}
            
            for material, data in accuracy_data.items():
                if len(data['actual']) > 0 and len(data['predicted']) > 0:
                    # Calculate MAPE
                    mape = mean_absolute_percentage_error(data['actual'], data['predicted'])
                    accuracy = max(0, 100 - mape)  # Convert to accuracy percentage
                    accuracies[material] = accuracy
                else:
                    accuracies[material] = 0.0
            
            return accuracies
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return {}
    
    def get_detailed_metrics(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get detailed performance metrics
        """
        try:
            accuracy_data = self.get_model_accuracy(project_id)
            
            metrics = []
            for material, accuracy in accuracy_data.items():
                metrics.append({
                    'material': material,
                    'accuracy': accuracy,
                    'status': 'Good' if accuracy >= 90 else 'Needs Improvement',
                    'trained': material in self.models_trained and self.models_trained[material]
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting detailed metrics: {str(e)}")
            return []
    
    def save_models(self, path: str = "models"):
        """
        Save trained models to disk
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            for material, model in self.models.items():
                model_path = os.path.join(path, f"{material.replace('/', '_')}_model.pkl")
                joblib.dump(model, model_path)
            
            # Save scalers and encoders
            joblib.dump(self.scalers, os.path.join(path, "scalers.pkl"))
            joblib.dump(self.encoders, os.path.join(path, "encoders.pkl"))
            
            logger.info(f"Models saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, path: str = "models"):
        """
        Load trained models from disk
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Models directory {path} not found")
                return
            
            # Load models
            for filename in os.listdir(path):
                if filename.endswith('_model.pkl'):
                    material = filename.replace('_model.pkl', '').replace('_', '/')
                    model_path = os.path.join(path, filename)
                    self.models[material] = joblib.load(model_path)
                    self.models_trained[material] = True
            
            # Load scalers and encoders
            scalers_path = os.path.join(path, "scalers.pkl")
            encoders_path = os.path.join(path, "encoders.pkl")
            
            if os.path.exists(scalers_path):
                self.scalers = joblib.load(scalers_path)
            
            if os.path.exists(encoders_path):
                self.encoders = joblib.load(encoders_path)
            
            logger.info(f"Loaded {len(self.models)} models from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
