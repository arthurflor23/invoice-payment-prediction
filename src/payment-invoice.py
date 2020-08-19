from utils import CSVManager

csv = CSVManager()
csv.read('InvoicedDocuments_v4.csv', orderby=['DocumentDate'])
csv.ppnan(dropna_cols=['ClearingDate'], fillna_value=0)

csv.cast_to_number(cols=['CustomerRegion', 'PaymentTerms'])
csv.cast_to_integer(cols=['InvoicedDocuments', 'PaidDocuments', 'PaidPastDocuments', 'OpenDocuments', 'PastDueDocuments'])
csv.cast_to_date(cols=['CustomerLastCreditReview', 'DocumentDate', 'DueDate', 'ClearingDate'])

csv.minmax_condition(min_cols=['DocumentDate', 'DocumentDate'], max_cols=['DueDate', 'ClearingDate'])
csv.extract_days(cols=['DocumentDate', 'DueDate'])

csv.set_ratio(dividend_cols=['InvoicedAmount', 'PaidAmount', 'PaidPastAmount', 'OpenAmount', 'PastDueAmount'],
              divisor_cols=['InvoicedDocuments', 'PaidDocuments', 'PaidPastDocuments', 'OpenDocuments', 'PastDueDocuments'])

csv.set_daysto(source_cols=['DocumentDate', 'DocumentDate', 'CustomerLastCreditReview'],
               target_cols=['DueDate', 'ClearingDate', 'DocumentDate'])

csv.set_range(bins=list(range(1, 31, 28)), cols=['DaysToDueDate', 'DaysToClearingDate'])

csv.calculate_per_bucket(bucket_col='DaysToClearingDateRangeCT', amount_col='DocumentAmount',
                         date_col='DocumentDate', key_col='CustomerKey', month_window=2)

csv.save('payment-invoice.csv')

# train, test = csv.get_data_range(col='DocumentDate', date='2020-07-01', month_window=2)

# y_column = np.array(['DaysToClearingDateRangeCT'])
# features = np.array(['CompanyKey',
#                      'CustomerKey',
#                      'CustomerRegion',
#                      'PaymentTerms',
#                      'DocumentAmount',
#                      'AvgDSOPastDueDocuments',
#                      'PastDueDays',
#                      'DaysToDueDate',
#                      'DocumentDateWeekDay',
#                      'DueDateWeekDay',
#                      'RatioInvoicedAmountInvoicedDocuments',
#                      'RatioPastDueAmountPastDueDocuments',
#                      'Bucket0Amount', 'Bucket0Count',
#                      'Bucket1Amount', 'Bucket1Count',
#                      ])

# x_train, y_train = train[features].values, train[y_column].values
# x_test, y_test = test[features].values, test[y_column].values

# random_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', min_weight_fraction_leaf=1e-4, random_state=42)
# random_forest.fit(x_train, np.squeeze(y_train))

# predict = random_forest.predict(x_test)

# print(f'Total items: {len(y_test)}')
# print(f'Accuracy: {accuracy_score(y_test, predict) * 100:.2f}%\n')
# print(confusion_matrix(y_test, predict))
