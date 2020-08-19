from utils import CSVManager

csv = CSVManager()
csv.read('InvoicedDocuments_v4.csv', orderby=['DocumentDate'])
csv.ppnan(dropna_cols=['ClearingDate'], fillna_value=0)

csv.cast_to_number(cols=['CustomerRegion', 'PaymentTerms'])
csv.cast_to_integer(cols=['InvoicedDocuments', 'PaidDocuments', 'PaidPastDocuments', 'OpenDocuments', 'PastDueDocuments'])
csv.cast_to_date(cols=['CustomerLastCreditReview', 'DocumentDate', 'DueDate', 'ClearingDate'])

csv.set_end_month(cols=['DueDate'])
csv.minmax_condition(min_cols=['ClearingDate'], max_cols=['DueDateEndMonth'], filt=False)

csv.minmax_condition(min_cols=['DocumentDate', 'DocumentDate'], max_cols=['DueDate', 'ClearingDate'])
csv.extract_days(cols=['DocumentDate', 'DueDate'])

csv.set_ratio(dividend_cols=['InvoicedAmount', 'PaidAmount', 'PaidPastAmount', 'OpenAmount', 'PastDueAmount'],
              divisor_cols=['InvoicedDocuments', 'PaidDocuments', 'PaidPastDocuments', 'OpenDocuments', 'PastDueDocuments'])

csv.set_daysto(source_cols=['DocumentDate', 'DocumentDate', 'DocumentDate', 'CustomerLastCreditReview'],
               target_cols=['DueDate', 'DueDateEndMonth', 'ClearingDate', 'DocumentDate'])

csv.calculate_per_bucket(bucket_col='AfterDueDateEndMonth', amount_col='DocumentAmount',
                         date_col='DocumentDate', key_col='CustomerKey', month_window=2)

csv.save('payment-invoice-month.csv')
