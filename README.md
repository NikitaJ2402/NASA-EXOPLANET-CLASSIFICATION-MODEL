# NASA-EXOPLANET-CLASSIFICATION-MODEL

This comprehensive machine learning project demonstrates how to build a complete ML pipeline using NASA exoplanet data, incorporating automated model selection (AutoML) techniques and achieving high-accuracy predictions for exoplanet classification.
Project Overview
The project creates a robust machine learning system that analyzes stellar observations to classify them as either confirmed exoplanets or false positives, mimicking the work done by NASA's Kepler space telescope mission. Using sophisticated data preprocessing, feature engineering, and automated model selection, the system achieves 94.38% accuracy with a Random Forest classifier.
Dataset Characteristics
The synthetic dataset is based on real NASA Kepler mission parameters and includes 8 key astronomical features:
1.	Stellar Mass - Mass of the host star in solar masses
2.	Stellar Radius - Radius of the host star in solar radii
3.	Stellar Temperature - Effective temperature of the star in Kelvin
4.	Orbital Period - Time for planet to complete one orbit in days
5.	Planet Radius - Size of the planet in Earth radii
6.	Transit Depth - Fraction of starlight blocked during transit
7.	Impact Parameter - Geometric parameter of the transit
8.	Transit Duration - Length of the transit event in hours
The final cleaned dataset contains 3,733 samples with a class distribution of 93.6% confirmed exoplanets and 6.4% false positives, reflecting the realistic challenge of exoplanet detection where most candidates are genuine discoveries.
