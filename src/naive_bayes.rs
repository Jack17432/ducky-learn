use std::collections::{HashMap, HashSet};
use super::util::{Fit, Unfit};

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
            self.probability_of_class.insert(uniq_class.to_string(), *y_counts.get(uniq_class).unwrap() as f64 / total_rows);

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

            self.probability_of_feat_by_class.insert(uniq_class.to_string(), class_feat_probs);
        }

        StdNaiveBayes{
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
                        let prob = self.probability_of_feat_by_class.get(class).unwrap().get(&j.to_string()).unwrap();
                        log_sum += (*feat_count * (*prob + small_number).ln());
                    }
                }
                row_probabilities.push(log_sum);
            }

            let max_prob_index = row_probabilities.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            y_pred.push(unique_classes[max_prob_index].to_string());
        }

        y_pred
    }
}


pub struct GaussianNaiveBayes<State = Unfit> {
    pub classes: Vec<String>,
    pub probability_of_class: HashMap<String, f64>,
    pub probability_of_feat_by_class: HashMap<String, (f64, f64)>,

    state: std::marker::PhantomData<State>,
}

impl GaussianNaiveBayes {
    pub fn new() -> Self {
        Self {
            classes: Default::default(),
            probability_of_class: Default::default(),
            probability_of_feat_by_class: Default::default(),

            state: Default::default(),
        }
    }

    pub fn fit(mut self, x: &Vec<Vec<f64>>, y: &Vec<String>) -> GaussianNaiveBayes<Fit> {
        let uniq_classes: Vec<String> = y.clone()
            .into_iter().collect::<HashSet<String>>()
            .into_iter().collect::<Vec<String>>();

        let fit_model = GaussianNaiveBayes{
            classes: uniq_classes.clone(),
            probability_of_class: calculate_class_probability(&uniq_classes, y),
            probability_of_feat_by_class: self.probability_of_feat_by_class.clone(),

            state: std::marker::PhantomData::<Fit>,
        };

        fit_model
    }
}

fn calculate_class_probability(uniq_classes: &Vec<String>, all_classes: &Vec<String>) -> HashMap<String, f64> {
    let mut class_probability: HashMap<String, f64> = HashMap::new();
    let total = all_classes.len() as f64;

    let mut class_counts: HashMap<&String, f64> = HashMap::new();

    // Calculate the counts for each class in one pass
    for class in all_classes {
        *class_counts.entry(class).or_insert(0.0) += 1.0;
    }

    // For each unique class, compute and store the probability
    uniq_classes.iter().map(|class| {
        let count = *class_counts.get(class).unwrap_or(&0.0);
        (class.clone(), count / total)
    }).collect()
}

fn calculate_feature_probability(x: &Vec<Vec<f64>>, y: &Vec<String>, uniq_classes: &Vec<String>) -> HashMap<String, Vec<(f64, f64)>> {
    let mut return_feature_prob: HashMap<String, Vec<(f64, f64)>> = HashMap::new();

    if x.len() != y.len() {
        return HashMap::new();
    }

    for class in uniq_classes {
        let x_class: Vec<_> = x.iter().zip(y)
            .filter_map(|(x, y)| if y == class {Some(x.clone())} else {None})
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
            let variance: f64 = feature_values.iter().map(|value| {
                let diff = mean - *value;
                diff * diff
            }).sum::<f64>() / feature_values.len() as f64;

            let std_dev = variance.sqrt();

            return_feature_prob.entry(class.to_string()).or_insert_with(|| Vec::with_capacity(num_features)).push((mean, std_dev));
        }
    }

    return_feature_prob
}

#[cfg(test)]
mod calculation_functions_tests {
    use super::*;

    #[test]
    fn test_calculate_class_probability() {
        let uniq_classes = vec!["class1".to_string(), "class2".to_string(), "class3".to_string()];
        let all_classes = vec!["class1".to_string(), "class2".to_string(), "class2".to_string(),
                               "class3".to_string(), "class3".to_string(), "class3".to_string()];
        let probabilities = calculate_class_probability(&uniq_classes, &all_classes);

        assert!(probabilities.get("class1").unwrap() - (1.0/6.0) < f64::EPSILON);
        assert!(probabilities.get("class2").unwrap() - (2.0/6.0) < f64::EPSILON);
        assert!(probabilities.get("class3").unwrap() - (3.0/6.0) < f64::EPSILON);
    }

    #[test]
    fn test_calculate_class_probability_sum_to_one() {
        let uniq_classes = vec!["class1".to_string(), "class2".to_string(), "class3".to_string()];
        let all_classes = vec!["class1".to_string(), "class2".to_string(), "class2".to_string(),
                               "class3".to_string(), "class3".to_string(), "class3".to_string()];
        let probabilities = calculate_class_probability(&uniq_classes, &all_classes);

        let sum: f64 = probabilities.values().sum();

        assert!(1.0 - sum < f64::EPSILON);
    }

    #[test]
    fn test_calculate_feature_probability() {
        let uniq_classes = vec!["class1".to_string(), "class2".to_string()];
        let y = vec!["class1".to_string(), "class2".to_string(), "class1".to_string(), "class2".to_string()];
        let x = vec![vec![1.0, 2.0], vec![2.0, 2.0], vec![2.0, 3.0], vec![3.0, 3.0]];

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
        let y = vec!["class1".to_string(), "class1".to_string(), "class2".to_string(), "class2".to_string()];
        let x = vec![vec![2.0, 2.0], vec![2.0, 2.0], vec![2.0, 2.0], vec![2.0, 2.0]];

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
}