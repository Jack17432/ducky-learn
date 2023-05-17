use std::collections::HashMap;
use std::collections::HashSet;

/// Marker struct indicating a `StdNaiveBayes` that has not been fit.
pub struct Unfit;

/// Marker struct indicating a `StdNaiveBayes` that has been fit.
pub struct Fit;

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
            probability_of_class: HashMap::new(),
            probability_of_feat_by_class: HashMap::new(),

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
