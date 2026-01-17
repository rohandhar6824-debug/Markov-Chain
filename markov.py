import numpy as np

class SimpleWeatherMarkov:
    """Simple Markov Chain for weather prediction"""
    
    def __init__(self):
        self.states = ['Sunny', 'Cloudy', 'Rainy']
        
        # Example transition matrix (based on real data)
        self.transition_matrix = np.array([
            [0.6, 0.3, 0.1],  # Sunny -> [Sunny, Cloudy, Rainy]
            [0.4, 0.4, 0.2],  # Cloudy -> [Sunny, Cloudy, Rainy]
            [0.2, 0.3, 0.5],  # Rainy -> [Sunny, Cloudy, Rainy]
        ])
        
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
    
    def predict(self, current_weather, days=1):
        """Predict next day's weather"""
        current_idx = self.state_to_idx[current_weather]
        forecast = [current_weather]
        
        for _ in range(days):
            # Get probabilities for next state
            probs = self.transition_matrix[current_idx]
            # Choose next state
            next_idx = np.random.choice(len(self.states), p=probs)
            next_weather = self.idx_to_state[next_idx]
            forecast.append(next_weather)
            current_idx = next_idx
        
        return forecast
    
    def get_probabilities(self, current_weather):
        """Get probabilities for next day's weather"""
        idx = self.state_to_idx[current_weather]
        return dict(zip(self.states, self.transition_matrix[idx]))


# Quick example
if __name__ == "__main__":
    weather_predictor = SimpleWeatherMarkov()
    
    print("Today's weather: Sunny")
    forecast = weather_predictor.predict('Sunny', 3)
    
    for i, weather in enumerate(forecast):
        day = "Today" if i == 0 else f"Day {i}"
        print(f"{day}: {weather}")
    
    print("\nTransition probabilities from Sunny:")
    probs = weather_predictor.get_probabilities('Sunny')
    for weather, prob in probs.items():
        print(f"  {weather}: {prob:.2f}")