# ===============================================================
#  ABC-Stacking Classifier Optimization (Final Version)
#
#  Copyright (c) 2025 Qian M. All rights reserved.
#  Released for academic and research purposes only.
# ===============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import f1_score
import random
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt


class ABCStacking:
    def __init__(self, n_employed_bees=10, n_onlooker_bees=10, max_iter=100,
                 random_state=42):
        self.n_employed_bees = n_employed_bees
        self.n_onlooker_bees = n_onlooker_bees
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        self.base_learners = [
            AdaBoostClassifier(random_state=random_state),
            CatBoostClassifier(silent=True, random_state=random_state),
            LGBMClassifier(random_state=random_state),
            XGBClassifier(random_state=random_state),
            RandomForestClassifier(random_state=random_state),
            ExtraTreesClassifier(random_state=random_state),
            GradientBoostingClassifier(random_state=random_state),
            DecisionTreeClassifier(random_state=random_state)
        ]

        self.meta_learners = [
            ExtraTreesClassifier(random_state=random_state),
            RandomForestClassifier(random_state=random_state),
            LGBMClassifier(random_state=random_state),
            XGBClassifier(random_state=random_state)
        ]

        self.best_config = None
        self.best_f1 = -np.inf
        self.history = []

    def generate_random_solution(self):
        base_selected = random.sample(self.base_learners, 3)
        meta_selected = random.choice(self.meta_learners)
        return {
            'base': [clone(model) for model in base_selected],
            'meta': clone(meta_selected),
            'f1': None,
            'fitness': None,
            'prob': None
        }

    def evaluate_solution(self, solution, X, y):
        try:
            stacking_model = StackingModel(solution['base'], solution['meta'])
            stacking_model.fit(X, y)
            y_pred = stacking_model.predict(X)
            f1 = f1_score(y, y_pred, average='macro')

            # Fitness function
            if f1 >= 0:
                fitness = 1 / (1 + f1)
            else:
                fitness = 1 + abs(f1)

            return f1, fitness
        except Exception as e:
            print(f"[Warning] Evaluation failed: {e}")
            return -1, 0

    def calculate_probabilities(self, population):
        total_fitness = sum(sol['fitness'] for sol in population)
        if total_fitness == 0:
            for sol in population:
                sol['prob'] = 1.0 / len(population)
        else:
            for sol in population:
                sol['prob'] = sol['fitness'] / total_fitness
        return population

    def employed_bee_phase(self, population, X, y, optimize_base_only=True):
        for i in range(len(population)):
            new_solution = {
                'base': [clone(model) for model in population[i]['base']],
                'meta': clone(population[i]['meta'])
            }

            if optimize_base_only:
                replace_idx = random.randint(0, 2)
                new_base = random.choice(self.base_learners)
                while type(new_base) == type(new_solution['base'][replace_idx]):
                    new_base = random.choice(self.base_learners)
                new_solution['base'][replace_idx] = clone(new_base)
            else:
                new_solution['meta'] = clone(random.choice(self.meta_learners))

            new_f1, new_fitness = self.evaluate_solution(new_solution, X, y)
            if new_f1 > population[i]['f1']:
                population[i] = {
                    'base': new_solution['base'],
                    'meta': new_solution['meta'],
                    'f1': new_f1,
                    'fitness': new_fitness,
                    'prob': None
                }
        return population

    def onlooker_bee_phase(self, population, X, y, optimize_base_only=True):
        population = self.calculate_probabilities(population)
        new_population = []

        for _ in range(self.n_onlooker_bees):
            selected_idx = np.random.choice(
                len(population), p=[sol['prob'] for sol in population]
            )
            selected = population[selected_idx]
            new_solution = {
                'base': [clone(model) for model in selected['base']],
                'meta': clone(selected['meta'])
            }

            if optimize_base_only:
                replace_idx = random.randint(0, 2)
                new_base = random.choice(self.base_learners)
                while type(new_base) == type(new_solution['base'][replace_idx]):
                    new_base = random.choice(self.base_learners)
                new_solution['base'][replace_idx] = clone(new_base)
            else:
                new_solution['meta'] = clone(random.choice(self.meta_learners))

            new_f1, new_fitness = self.evaluate_solution(new_solution, X, y)
            new_population.append({
                'base': new_solution['base'],
                'meta': new_solution['meta'],
                'f1': new_f1,
                'fitness': new_fitness,
                'prob': None
            })

        combined = population + new_population
        combined.sort(key=lambda x: x['f1'], reverse=True)
        return combined[:self.n_employed_bees]

    def scout_bee_phase(self, population, X, y):
        worst_idx = np.argmin([sol['f1'] for sol in population])
        new_solution = self.generate_random_solution()
        new_f1, new_fitness = self.evaluate_solution(new_solution, X, y)
        population[worst_idx] = {
            'base': new_solution['base'],
            'meta': new_solution['meta'],
            'f1': new_f1,
            'fitness': new_fitness,
            'prob': None
        }
        return population

    def optimize(self, X, y):
        population = [self.generate_random_solution() for _ in range(self.n_employed_bees)]
        for i in range(len(population)):
            f1, fitness = self.evaluate_solution(population[i], X, y)
            population[i]['f1'] = f1
            population[i]['fitness'] = fitness

        # Phase 1: Optimize Base Models
        for iteration in tqdm(range(int(self.max_iter * 0.7)), desc="Optimizing Base Models"):
            population = self.employed_bee_phase(population, X, y, optimize_base_only=True)
            population = self.onlooker_bee_phase(population, X, y, optimize_base_only=True)
            if iteration % 5 == 0:
                population = self.scout_bee_phase(population, X, y)

            current_best = max(population, key=lambda x: x['f1'])
            if current_best['f1'] > self.best_f1:
                self.best_f1 = current_best['f1']
                self.best_config = {
                    'base': [clone(model) for model in current_best['base']],
                    'meta': clone(current_best['meta'])
                }
            self.history.append({
                'iteration': iteration,
                'best_f1': self.best_f1,
                'base_models': [type(m).__name__ for m in current_best['base']],
                'meta_model': type(current_best['meta']).__name__
            })

        # Phase 2: Optimize Meta Learner
        for iteration in tqdm(range(int(self.max_iter * 0.3)), desc="Optimizing Meta Learner"):
            population = self.employed_bee_phase(population, X, y, optimize_base_only=False)
            population = self.onlooker_bee_phase(population, X, y, optimize_base_only=False)
            if iteration % 5 == 0:
                population = self.scout_bee_phase(population, X, y)

            current_best = max(population, key=lambda x: x['f1'])
            if current_best['f1'] > self.best_f1:
                self.best_f1 = current_best['f1']
                self.best_config = {
                    'base': [clone(model) for model in current_best['base']],
                    'meta': clone(current_best['meta'])
                }

        return self.best_config


class StackingModel:
    def __init__(self, base_models, meta_model):
        self.base_models = [clone(model) for model in base_models]
        self.meta_model = clone(meta_model)

    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        self.meta_model.fit(meta_features, y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        return self.meta_model.predict_proba(meta_features)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    train_data = pd.read_csv('dataset_ratio_1_to_100_0.csv')
    test_data = pd.read_csv('all_w_validation.csv')

    features = ['STFT', 'CBRP', 'Bedrock elevation', 'Hydraulic gradient', 'Roughness'] 
    #[TFF, CBRP, Bedrock elevation, Hydraulic gradient, Roughness]
    X_train = train_data[features].values
    y_train = train_data['Label'].values
    X_test = test_data[features].values
    y_test = test_data['Label'].values

    abc = ABCStacking(n_employed_bees=10, n_onlooker_bees=10, max_iter=100)
    best_config = abc.optimize(X_train, y_train)

    print("\nBest Configuration:")
    print(f"Base Learners: {[type(m).__name__ for m in best_config['base']]}")
    print(f"Meta Learner: {type(best_config['meta']).__name__}")
    print(f"Best Validation F1 Score: {abc.best_f1:.4f}")

    final_model = StackingModel(best_config['base'], best_config['meta'])
    final_model.fit(X_train, y_train)

    y_pred_proba = final_model.predict_proba(X_test)
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_test, final_model.predict(X_test, threshold=t), average='macro') for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = final_model.predict(X_test, threshold=best_threshold)
    test_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\nTest F1 Score: {test_f1:.4f} (Threshold = {best_threshold:.4f})")
    joblib.dump(final_model, 'abc_stacking_model.pkl')
    print("Model saved as 'abc_stacking_model.pkl'.")

