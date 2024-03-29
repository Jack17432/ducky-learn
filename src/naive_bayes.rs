use super::util::{Fit, Unfit};
use std::collections::{HashMap, HashSet};

/// Implementation of a standard Naive Bayes classifier.
///
/// This classifier uses Laplace smoothing, the degree of which can be controlled with the `alpha` parameter.
///
/// # Parameters
/// - `alpha`: The Laplace smoothing factor.
/// - `probability_of_class`: HashMap storing the probabilities of each class.
/// - `probability_of_feat_by_class`: HashMap storing the probabilities of each feature given a class.
/// - `state`: PhantomData indicating whether the classifier has been fit.
///
/// # Type parameters
/// - `State`: Indicates whether the classifier has been fit. Can either be `Fit` or `Unfit`.
///
/// # Example
///
/// ```
/// use ducky_learn::naive_bayes::StdNaiveBayes;
///
/// // Define train and test data
/// let x_train: Vec<Vec<f64>> = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![2.0, 3.0, 4.0],
///     vec![3.0, 4.0, 5.0],
/// ];
/// let y_train: Vec<String> = vec!["class1".to_string(), "class2".to_string(), "class1".to_string()];
///
/// let x_test: Vec<Vec<f64>> = vec![
///     vec![1.5, 2.5, 3.5],
///     vec![2.5, 3.5, 4.5],
/// ];
///
/// let mut nb = StdNaiveBayes::new(1.0);
/// let nb = nb.fit(&x_train, &y_train);
/// let y_pred = nb.predict(&x_test);
///
/// // y_pred will hold the predicted classes for x_test
/// ```
#[derive(Debug)]
pub struct StdNaiveBayes<State = Unfit> {
    pub alpha: f64,
    pub probability_of_class: HashMap<String, f64>,
    pub probability_of_feat_by_class: HashMap<String, HashMap<String, f64>>,

    state: std::marker::PhantomData<State>,
}

impl StdNaiveBayes {
    /// Constructs a new, unfitted `StdNaiveBayes` classifier with a specified alpha value.
    ///
    /// # Parameters
    /// - `alpha`: The Laplace smoothing factor.
    ///
    /// # Returns
    /// A new `StdNaiveBayes` instance.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            probability_of_class: Default::default(),
            probability_of_feat_by_class: Default::default(),

            state: Default::default(),
        }
    }

    /// Fits the `StdNaiveBayes` classifier to the training data.
    ///
    /// # Parameters
    /// - `x`: The training data.
    /// - `y`: The target values.
    ///
    /// # Returns
    /// The fitted `StdNaiveBayes` classifier.
    pub fn fit(mut self, x: &Vec<Vec<f64>>, y: &Vec<String>) -> StdNaiveBayes<Fit> {
        let mut y_counts: HashMap<String, i32> = HashMap::new();
        for class in y {
            let counter = y_counts.entry(class.to_string()).or_insert(0);
            *counter += 1;
        }

        let total_rows = y.len() as f64;
        let unique_classes: HashSet<String> = y.into_iter().cloned().collect();

        for uniq_class in &unique_classes {
            self.probability_of_class.insert(
                uniq_class.to_string(),
                *y_counts.get(uniq_class).unwrap() as f64 / total_rows,
            );

            let mut class_feat_probs: HashMap<String, f64> = HashMap::new();
            let mut sum_of_feats_in_class = 0.0;
            for (i, class) in y.iter().enumerate() {
                if class == uniq_class {
                    for (j, feat_count) in x[i].iter().enumerate() {
                        let counter = class_feat_probs.entry(j.to_string()).or_insert(0.0);
                        *counter += *feat_count;
                        sum_of_feats_in_class += *feat_count;
                    }
                }
            }
            sum_of_feats_in_class += self.alpha * x[0].len() as f64;

            for (feat, count) in class_feat_probs.iter_mut() {
                *count = (*count + self.alpha) / sum_of_feats_in_class;
            }

            self.probability_of_feat_by_class
                .insert(uniq_class.to_string(), class_feat_probs);
        }

        StdNaiveBayes {
            alpha: self.alpha,
            probability_of_class: self.probability_of_class.clone(),
            probability_of_feat_by_class: self.probability_of_feat_by_class.clone(),

            state: std::marker::PhantomData::<Fit>,
        }
    }
}

impl StdNaiveBayes<Fit> {
    /// Predicts the target values for the given data.
    ///
    /// # Parameters
    /// - `x`: The data to predict target values for.
    ///
    /// # Returns
    /// The predicted target values.
    ///
    /// # Panics
    /// This function will panic if the classifier has not been fit.
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<String> {
        let mut y_pred: Vec<String> = Vec::new();
        let unique_classes: Vec<String> = self.probability_of_class.keys().cloned().collect();
        let class_probabilities: Vec<f64> = self.probability_of_class.values().cloned().collect();
        let small_number = 1e-9;

        for row in x {
            let mut row_probabilities: Vec<f64> = Vec::new();
            for (i, class) in unique_classes.iter().enumerate() {
                let mut log_sum = (class_probabilities[i] + small_number).ln();
                for (j, feat_count) in row.iter().enumerate() {
                    if *feat_count > 0.0 {
                        let prob = self
                            .probability_of_feat_by_class
                            .get(class)
                            .unwrap()
                            .get(&j.to_string())
                            .unwrap();
                        log_sum += (*feat_count * (*prob + small_number).ln());
                    }
                }
                row_probabilities.push(log_sum);
            }

            let max_prob_index = row_probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            y_pred.push(unique_classes[max_prob_index].to_string());
        }

        y_pred
    }
}

/// The `GaussianNaiveBayes` struct represents a Gaussian Naive Bayes classifier.
///
/// A Gaussian Naive Bayes classifier is a type of probabilistic machine learning model
/// used for classification tasks. The Gaussian version assumes the features that it is
/// learning from are distributed normally.
///
/// This struct has two possible states: `Unfit` and `Fit`. An `Unfit` model is one
/// that has not yet been trained on data, while a `Fit` model has been trained and
/// can be used for making predictions.
///
/// # Fields
///
/// * `classes` - A vector of unique class labels (targets) that the model may predict.
///
/// * `probability_of_class` - A hashmap where keys are the class labels and the values
///   are the corresponding prior probabilities of each class.
///
/// * `probability_of_feat_by_class` - A hashmap where keys are the class labels and
///   the values are vectors of tuples. Each tuple represents the mean and standard
///   deviation of a particular feature for that class.
///
/// * `state` - A marker for the model's state. This is either `Unfit` (for a newly
///   instantiated model) or `Fit` (for a model that has been trained on data).
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use ducky_learn::naive_bayes::GaussianNaiveBayes;
///
/// let model = GaussianNaiveBayes::new();
///
/// let x_train: Vec<Vec<f64>> = vec![
///     vec![1.0, 2.0],
///     vec![2.0, 3.0],
///     vec![3.0, 4.0],
///     vec![4.0, 5.0],
/// ];
/// let y_train: Vec<String> = vec![
///     "class1".to_string(),
///     "class2".to_string(),
///     "class1".to_string(),
///     "class2".to_string(),
/// ];
///
/// let model = model.fit(&x_train, &y_train);
///
/// let x_test: Vec<Vec<f64>> = vec![
///     vec![1.5, 2.5],
///     vec![3.5, 4.5],
/// ];
///
/// let predictions = model.predict(&x_test);
///
/// println!("{:?}", predictions);
/// ```
pub struct GaussianNaiveBayes<State = Unfit> {
    pub classes: Vec<String>,
    pub probability_of_class: HashMap<String, f64>,
    pub probability_of_feat_by_class: HashMap<String, Vec<(f64, f64)>>,

    state: std::marker::PhantomData<State>,
}

impl GaussianNaiveBayes {
    /// Creates a new `GaussianNaiveBayes` instance with an `Unfit` state.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of `GaussianNaiveBayes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::naive_bayes::GaussianNaiveBayes;
    /// let model = GaussianNaiveBayes::new();
    /// ```
    pub fn new() -> Self {
        Self {
            classes: Default::default(),
            probability_of_class: Default::default(),
            probability_of_feat_by_class: Default::default(),

            state: Default::default(),
        }
    }

    /// Fits the model on the provided dataset, updating the model's state to `Fit`.
    ///
    /// # Arguments
    ///
    /// * `x` - A reference to a vector of vectors, where each inner vector represents
    ///   the features of a data point.
    ///
    /// * `y` - A reference to a vector of class labels for each data point in `x`.
    ///
    /// # Returns
    ///
    /// * `GaussianNaiveBayes<Fit>` - The same model instance with updated fields
    ///   and state set to `Fit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::naive_bayes::GaussianNaiveBayes;
    ///
    /// let model = GaussianNaiveBayes::new();
    ///
    /// let x_train: Vec<Vec<f64>> = vec![
    ///     vec![1.0, 2.0],
    ///     vec![2.0, 3.0],
    ///     vec![3.0, 4.0],
    ///     vec![4.0, 5.0],
    /// ];
    /// let y_train: Vec<String> = vec![
    ///     "class1".to_string(),
    ///     "class2".to_string(),
    ///     "class1".to_string(),
    ///     "class2".to_string(),
    /// ];
    ///
    /// let model = model.fit(&x_train, &y_train);
    /// ```
    pub fn fit(mut self, x: &Vec<Vec<f64>>, y: &Vec<String>) -> GaussianNaiveBayes<Fit> {
        let uniq_classes: Vec<String> = y
            .clone()
            .into_iter()
            .collect::<HashSet<String>>()
            .into_iter()
            .collect::<Vec<String>>();

        GaussianNaiveBayes {
            probability_of_class: calculate_class_probability(&uniq_classes, y),
            probability_of_feat_by_class: calculate_feature_probability(x, y, &uniq_classes),
            classes: uniq_classes,

            state: std::marker::PhantomData::<Fit>,
        }
    }
}

impl GaussianNaiveBayes<Fit> {
    /// Predicts the class of the provided data points.
    ///
    /// # Arguments
    ///
    /// * `x` - A reference to a vector of vectors, where each inner vector represents
    ///   the features of a data point.
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - A vector of predicted class labels for each data point in `x`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::naive_bayes::GaussianNaiveBayes;
    ///
    /// let model = GaussianNaiveBayes::new().fit(
    ///     &vec![vec![0.1, 0.5], vec![0.6, 0.6]],
    ///     &vec!["class1".to_string(), "class2".to_string()]
    /// );
    ///
    /// let x_test: Vec<Vec<f64>> = vec![
    ///     vec![1.5, 2.5],
    ///     vec![3.5, 4.5],
    /// ];
    ///
    /// let predictions = model.predict(&x_test);
    ///
    /// println!("{:?}", predictions);
    /// ```
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<String> {
        let mut predictions: Vec<String> = Vec::new();

        for data in x.iter() {
            let mut max_prob = f64::NEG_INFINITY;
            let mut max_class = String::from("");

            for class in &self.classes {
                let mut class_prob = self.probability_of_class.get(class).unwrap().ln();

                if let Some(feature_probs) = self.probability_of_feat_by_class.get(class) {
                    for (index, &(mean, std_dev)) in feature_probs.iter().enumerate() {
                        let feature_value = data[index];
                        let feature_prob = calculate_probability(feature_value, mean, std_dev);
                        class_prob += feature_prob.ln();
                    }
                }

                if class_prob > max_prob {
                    max_prob = class_prob;
                    max_class = class.clone();
                }
            }
            predictions.push(max_class);
        }

        predictions
    }
}

fn calculate_mean(data: &Vec<f64>) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

fn calculate_std_dev(data: &Vec<f64>, mean: f64) -> f64 {
    let variance: f64 = data
        .iter()
        .map(|&value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;

    variance.sqrt()
}

fn calculate_probability(x: f64, mean: f64, std_dev: f64) -> f64 {
    let exponent = (-((x - mean).powi(2)) / (2.0 * std_dev.powi(2))).exp();
    (1.0 / (2.0 * std::f64::consts::PI * std_dev.powi(2)).sqrt()) * exponent
}

fn calculate_class_probability(
    uniq_classes: &Vec<String>,
    all_classes: &Vec<String>,
) -> HashMap<String, f64> {
    let mut class_probability: HashMap<String, f64> = HashMap::new();
    let total = all_classes.len() as f64;

    let mut class_counts: HashMap<&String, f64> = HashMap::new();

    // Calculate the counts for each class in one pass
    for class in all_classes {
        *class_counts.entry(class).or_insert(0.0) += 1.0;
    }

    // For each unique class, compute and store the probability
    uniq_classes
        .iter()
        .map(|class| {
            let count = *class_counts.get(class).unwrap_or(&0.0);
            (class.clone(), count / total)
        })
        .collect()
}

fn calculate_feature_probability(
    x: &Vec<Vec<f64>>,
    y: &Vec<String>,
    uniq_classes: &Vec<String>,
) -> HashMap<String, Vec<(f64, f64)>> {
    let mut return_feature_prob: HashMap<String, Vec<(f64, f64)>> = HashMap::new();

    if x.len() != y.len() {
        return HashMap::new();
    }

    for class in uniq_classes {
        let x_class: Vec<_> = x
            .iter()
            .zip(y)
            .filter_map(|(x, y)| if y == class { Some(x.clone()) } else { None })
            .collect();

        if x_class.is_empty() {
            continue;
        }

        let num_features = x_class[0].len();

        for i in 0..num_features {
            let feature_values: Vec<_> = x_class.iter().map(|features| features[i]).collect();

            // calculate the mean
            let mean: f64 = feature_values.iter().sum::<f64>() / feature_values.len() as f64;

            // calculate the standard deviation
            let variance: f64 = feature_values
                .iter()
                .map(|value| {
                    let diff = mean - *value;
                    diff * diff
                })
                .sum::<f64>()
                / feature_values.len() as f64;

            let std_dev = variance.sqrt();

            return_feature_prob
                .entry(class.to_string())
                .or_insert_with(|| Vec::with_capacity(num_features))
                .push((mean, std_dev));
        }
    }

    return_feature_prob
}

#[cfg(test)]
mod calculation_functions_tests {
    use super::*;

    #[test]
    fn test_calculate_class_probability() {
        let uniq_classes = vec![
            "class1".to_string(),
            "class2".to_string(),
            "class3".to_string(),
        ];
        let all_classes = vec![
            "class1".to_string(),
            "class2".to_string(),
            "class2".to_string(),
            "class3".to_string(),
            "class3".to_string(),
            "class3".to_string(),
        ];
        let probabilities = calculate_class_probability(&uniq_classes, &all_classes);

        assert!(probabilities.get("class1").unwrap() - (1.0 / 6.0) < f64::EPSILON);
        assert!(probabilities.get("class2").unwrap() - (2.0 / 6.0) < f64::EPSILON);
        assert!(probabilities.get("class3").unwrap() - (3.0 / 6.0) < f64::EPSILON);
    }

    #[test]
    fn test_calculate_class_probability_sum_to_one() {
        let uniq_classes = vec![
            "class1".to_string(),
            "class2".to_string(),
            "class3".to_string(),
        ];
        let all_classes = vec![
            "class1".to_string(),
            "class2".to_string(),
            "class2".to_string(),
            "class3".to_string(),
            "class3".to_string(),
            "class3".to_string(),
        ];
        let probabilities = calculate_class_probability(&uniq_classes, &all_classes);

        let sum: f64 = probabilities.values().sum();

        assert!(1.0 - sum < f64::EPSILON);
    }

    #[test]
    fn test_calculate_feature_probability() {
        let uniq_classes = vec!["class1".to_string(), "class2".to_string()];
        let y = vec![
            "class1".to_string(),
            "class2".to_string(),
            "class1".to_string(),
            "class2".to_string(),
        ];
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 3.0],
        ];

        let feature_probabilities = calculate_feature_probability(&x, &y, &uniq_classes);

        let class1_probabilities = feature_probabilities.get("class1").unwrap();
        assert!((class1_probabilities[0].0 - 1.5).abs() < f64::EPSILON);
        assert!((class1_probabilities[0].1 - 0.5).abs() < f64::EPSILON);
        assert!((class1_probabilities[1].0 - 2.5).abs() < f64::EPSILON);
        assert!((class1_probabilities[1].1 - 0.5).abs() < f64::EPSILON);

        let class2_probabilities = feature_probabilities.get("class2").unwrap();
        assert!((class2_probabilities[0].0 - 2.5).abs() < f64::EPSILON);
        assert!((class2_probabilities[0].1 - 0.5).abs() < f64::EPSILON);
        assert!((class2_probabilities[1].0 - 2.5).abs() < f64::EPSILON);
        assert!((class2_probabilities[1].1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_feature_probability_no_data() {
        let uniq_classes = vec!["class1".to_string(), "class2".to_string()];
        let y = vec![];
        let x = vec![];

        let feature_probabilities = calculate_feature_probability(&x, &y, &uniq_classes);

        assert!(feature_probabilities.is_empty());
    }

    #[test]
    fn test_calculate_feature_probability_same_feature_values() {
        let uniq_classes = vec!["class1".to_string(), "class2".to_string()];
        let y = vec![
            "class1".to_string(),
            "class1".to_string(),
            "class2".to_string(),
            "class2".to_string(),
        ];
        let x = vec![
            vec![2.0, 2.0],
            vec![2.0, 2.0],
            vec![2.0, 2.0],
            vec![2.0, 2.0],
        ];

        let feature_probabilities = calculate_feature_probability(&x, &y, &uniq_classes);

        let class1_probabilities = feature_probabilities.get("class1").unwrap();
        assert!((class1_probabilities[0].0 - 2.0).abs() < f64::EPSILON);
        assert!((class1_probabilities[0].1 - 0.0).abs() < f64::EPSILON);
        assert!((class1_probabilities[1].0 - 2.0).abs() < f64::EPSILON);
        assert!((class1_probabilities[1].1 - 0.0).abs() < f64::EPSILON);

        let class2_probabilities = feature_probabilities.get("class2").unwrap();
        assert!((class2_probabilities[0].0 - 2.0).abs() < f64::EPSILON);
        assert!((class2_probabilities[0].1 - 0.0).abs() < f64::EPSILON);
        assert!((class2_probabilities[1].0 - 2.0).abs() < f64::EPSILON);
        assert!((class2_probabilities[1].1 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_feature_probability_mismatched_lengths() {
        let uniq_classes = vec!["class1".to_string(), "class2".to_string()];
        let y = vec!["class1".to_string(), "class2".to_string()];
        let x = vec![];

        let feature_probabilities = calculate_feature_probability(&x, &y, &uniq_classes);

        assert!(feature_probabilities.is_empty());
    }

    #[test]
    fn test_calculate_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_mean(&data), 3.0);
    }

    #[test]
    fn test_calculate_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = calculate_mean(&data);
        assert_eq!(
            (calculate_std_dev(&data, mean) - 1.414213).abs() < 0.00001,
            true
        );
    }

    #[test]
    fn test_calculate_probability() {
        let x = 2.0;
        let mean = 2.0;
        let std_dev = 1.0;
        assert_eq!(
            (calculate_probability(x, mean, std_dev) - 0.398942).abs() < 0.00001,
            true
        );
    }
}

#[cfg(test)]
mod naive_bayes_tests {
    use super::*;

    #[test]
    fn test_fit_std() {
        let mut model = StdNaiveBayes::new(1.0);

        let x: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 1.0],
            vec![3.0, 1.0, 2.0],
        ];

        let y: Vec<String> = vec![
            "class1".to_string(),
            "class2".to_string(),
            "class1".to_string(),
        ];

        let model = model.fit(&x, &y);

        assert!((model.probability_of_class.get("class1").unwrap() - 2.0 / 3.0).abs() < 1e-9);
        assert!((model.probability_of_class.get("class2").unwrap() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_predict_std() {
        let mut model = StdNaiveBayes::new(1.0);

        let x: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0, 1.0, 2.0],
            vec![2.0, 3.0, 4.0, 2.0, 3.0],
            vec![4.0, 4.0, 5.0, 4.0, 4.0],
            vec![5.0, 5.0, 6.0, 5.0, 5.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ];

        let y: Vec<String> = vec![
            "class1".to_string(),
            "class1".to_string(),
            "class2".to_string(),
            "class2".to_string(),
            "class1".to_string(),
        ];

        let model = model.fit(&x, &y);

        let x_test: Vec<Vec<f64>> =
            vec![vec![1.5, 2.5, 3.5, 1.5, 2.5], vec![5.5, 4.5, 5.5, 4.5, 4.5]];

        let predictions = model.predict(&x_test);

        assert_eq!(predictions, vec!["class1", "class2"]);
    }

    #[test]
    fn test_new_gaus() {
        let model: GaussianNaiveBayes = GaussianNaiveBayes::new();

        assert_eq!(model.classes.len(), 0);
        assert_eq!(model.probability_of_class.len(), 0);
        assert_eq!(model.probability_of_feat_by_class.len(), 0);
    }

    #[test]
    fn test_fit_gaus() {
        let mut model: GaussianNaiveBayes = GaussianNaiveBayes::new();
        let x = vec![
            vec![2.0, 1.0],
            vec![3.0, 2.0],
            vec![2.5, 1.5],
            vec![4.0, 3.0],
        ];
        let y = vec![
            "class1".to_string(),
            "class1".to_string(),
            "class2".to_string(),
            "class2".to_string(),
        ];
        let model = model.fit(&x, &y);

        assert_eq!(model.classes.len(), 2);
        assert!(model.classes.contains(&"class1".to_string()));
        assert!(model.classes.contains(&"class2".to_string()));

        assert_eq!(model.probability_of_class.len(), 2);
        assert!(model
            .probability_of_class
            .contains_key(&"class1".to_string()));
        assert!(model
            .probability_of_class
            .contains_key(&"class2".to_string()));

        assert_eq!(model.probability_of_feat_by_class.len(), 2);
        assert!(model
            .probability_of_feat_by_class
            .contains_key(&"class1".to_string()));
        assert!(model
            .probability_of_feat_by_class
            .contains_key(&"class2".to_string()));
    }

    #[test]
    fn test_predict_gaus() {
        let mut model: GaussianNaiveBayes = GaussianNaiveBayes::new();
        let x = vec![
            vec![2.0, 1.0],
            vec![3.0, 2.0],
            vec![2.5, 1.5],
            vec![4.0, 3.0],
        ];
        let y = vec![
            "class1".to_string(),
            "class1".to_string(),
            "class2".to_string(),
            "class2".to_string(),
        ];
        let model = model.fit(&x, &y);

        let x_test = vec![vec![2.0, 1.0], vec![4.0, 3.0]];

        let predictions = model.predict(&x_test);
        assert_eq!(predictions.len(), x_test.len());
        assert_eq!(predictions[0], "class1");
        assert_eq!(predictions[1], "class2");
    }
}
