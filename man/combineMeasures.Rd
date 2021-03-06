% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/calibration.r
\name{combineMeasures}
\alias{combineMeasures}
\title{Combine all dataframes of the calibration performance measures}
\usage{
combineMeasures(y_test, algorithm, probMat, prev_result = NULL)
}
\arguments{
\item{y_test}{vector of true class of instances}

\item{algorithm}{give the name of the algorithm you used e.g. "Platt1"}

\item{probMat}{a matrix with each row being a probability distribution of the classes from the test data}

\item{prev_result}{pass dataframe returned by combineMeasures() to combine several results into one dataframe if any}
}
\value{
a dataframe that lists all metrics for the given test labels and the calibrated probability matrix
}
\description{
the function extends getCalibMeasures to handle multi-class case when
the test label has more than 2 classes. It also combines results
from combineMeasures() for comparing different algorithms by passing the
output of combinMeasures() to the argument called "prev_result" in the
function
}
