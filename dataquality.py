# Databricks notebook source
# DBTITLE 1,update & install package
!pip install --upgrade pip
!pip install great_expectations==1.0.5

# COMMAND ----------

# DBTITLE 1,import packages
import re
import great_expectations as gx
import pandas as pd
from datetime import datetime as dt
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window
from typing import Literal
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType, StructField, ArrayType, TimestampType, BooleanType, LongType

print(f'Great Expectation version {gx.__version__} is installed')

# COMMAND ----------

# DBTITLE 1,get constant
# MAGIC %run ./const

# COMMAND ----------

# DBTITLE 1,class ExpectationGenerator
class ExpectationGenerator:
    """
    A class to generate Great Expectations expectations based on provided rules.

    Attributes:
    - rule (dict): A dictionary containing the rule details.
    - fieldName (str): The name of the field/column to which the rule applies.
    - meta (dict): Metadata for the expectation including the expectation type and action.

    Methods:
    - required(): Generates an expectation that a column must exist.
    - choice(value_set: str): Generates an expectation that column values must be within a specified set.
    - date_format(strftime_format: str): Generates an expectation that column values match a specific date format.
    - time_format(strftime_format: str): Generates an expectation that column values match a specific time format.
    - string_format(regex: str): Generates an expectation that column values match a specific regex pattern.
    - data_type(expected_type: Literal['string', 'int', 'decimal', 'boolean', 'bit']): Generates an expectation that column values are of a specific data type.
    - number_range(min_value: int|float, max_value: int|float): Generates an expectation that column values are within a specified numeric range.
    - not_future_datetime(): Generates an expectation that datetime values are not in the future.
    - not_future_date(): Generates an expectation that date values are not in the future.
    - max_length(max_value: int|float): Generates an expectation that the length of column values does not exceed a maximum value.
    - min_length(min_value: int|float): Generates an expectation that the length of column values is at least a minimum value.
    - string_validation(str_contains: str): Generates an expectation that column values contain a specific substring.
    """
    def __init__(self, rule: dict) -> 'ExpectationGenerator':
        self.rule = rule
        self.fieldName = rule['field_name']
        self.meta = dict(
            expectation=rule['expectation'],
            action=rule['expectation_action'],
            operator=rule['operator']
        )
        
    def required(self):
        return gx.expectations.ExpectColumnValuesToNotBeNull(
            column=self.fieldName,
            meta = self.meta
        )
    
    def choice(self, *value_set):
        value_set = list(value_set)
        return gx.expectations.ExpectColumnValuesToBeInSet(
            column=self.fieldName,
            value_set=value_set,
            meta = self.meta
        )
    
    def date_format(self, strftime_format: str):
        return gx.expectations.ExpectColumnValuesToMatchStrftimeFormat(
            column=self.fieldName,
            strftime_format=strftime_format,
            meta = self.meta
        )
    
    def time_format(self, strftime_format: str):
        return gx.expectations.ExpectColumnValuesToMatchStrftimeFormat(
            column=self.fieldName,
            strftime_format=strftime_format,
            meta = self.meta
        )
    
    def string_format(self, regex: str):
        return gx.expectations.ExpectColumnValuesToMatchRegex(
            column=self.fieldName,
            regex=regex,
            meta = self.meta
        )

    def data_type(self, expected_type: Literal['string', 'int', 'decimal', 'boolean', 'bit']):
        type_map = dict(
            string='StringType',
            int='IntegerType',
            decimal='DoubleType',
            boolean='BooleanType',
            bit='BooleanType'
        )

        type_ = type_map[expected_type]

        return gx.expectations.ExpectColumnValuesToBeOfType(
            column=self.fieldName,
            type_=type_,
            meta = self.meta
        )
    
    def number_range(self, min_value: str, max_value: str):
        min_value = float(min_value)
        max_value = float(max_value)
        
        return gx.expectations.ExpectColumnValuesToBeBetween(
            column=self.fieldName,
            min_value=min_value,
            max_value=max_value,
            meta = self.meta
        )
    
    def not_future_datetime(self):
        now = str(dt.now())
        return gx.expectations.ExpectColumnValuesToBeBetween(
            column=self.fieldName, 
            max_value=now,
            meta = self.meta
        )
    
    def not_future_date(self):
        today = str(dt.today().replace(hour=0, minute=0, second=0))
        return gx.expectations.ExpectColumnValuesToBeBetween(
            column=self.fieldName, 
            max_value=today,
            meta = self.meta
        )

    def max_length(self, max_value: str):
        max_value = float(max_value)
        return gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column=self.fieldName, 
            max_value=max_value, 
            meta = self.meta
        )
    
    def min_length(self, min_value: str):
        min_value = float(min_value)
        return gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column=self.fieldName, 
            min_value=min_value, 
            meta = self.meta
        )

    def string_validation(self, str_contains: str):
        regex = f'.*{str_contains}.*'
        return gx.expectations.ExpectColumnValuesToMatchRegex(
            column=self.fieldName,
            regex=regex,
            meta = self.meta
        )

# COMMAND ----------

# DBTITLE 1,def create_expectation
def create_expectation(rule: dict) -> gx.core.ExpectationSuiteValidationResult:
    """
    Creates an expectation based on a rule.

    Args:
    - rule (dict): The rule dictionary.

    Returns:
    - gx.core.ExpectationSuiteValidationResult: The expectation object.
    """
    generator = ExpectationGenerator(rule)
    raw_expectation = rule['expectation_single'].strip()
    pattern = r'\((.*?)\)'
    match = re.search(pattern, raw_expectation)

    if match:
        args = [arg.replace("'", "").strip() for arg in match.group(1).split(",")]

        expectation = raw_expectation.split('(')[0]
        func = getattr(generator, expectation, None)

        return func(*args)
    
    else:
        func = getattr(generator, raw_expectation, None)
        return func()

# COMMAND ----------

# DBTITLE 1,def create_expectation_suite
def _get_rule_components(expectation) -> list[str]:
    # Use regex to split by 'and' and 'or' and keep them in the result
    components = re.split(r'(\s+or\s+|\s+and\s+)', expectation)
    
    # Remove any surrounding spaces from the results
    items = []
    for item in components:
        strip_item = item.strip()

        items.append(strip_item)

        # If expectation is required, then evaluate as required + min_length(1)
        if strip_item == 'required':
            items.append('and')
            items.append('min_length(1)')
    
    return items

def create_expectation_suite(
    context: gx.data_context.AbstractDataContext,
    suite_name: str = "expectation suite", 
    rules: list = []
) -> gx.ExpectationSuite:
    """
    Curate valid expectations and create an expectation suite.

    Args:
    - context(AbstractDataContext): The Great Expectation context object.
    - suite_name (str): The name of the expectation suite.
    - rules (list): A list of rules.

    Returns:
    - gx.ExpectationSuite: The expectation suite object.
    """
    pattern = r"(required|not_future_date|not_future_datetime|choice|date_format|time_format|string_format|data_type|number_range|min_length|max_length|string_validation)(\(([^\)]+)\))?(\s+and\s+|\s+or\s+)?"

    rules = [rule for rule in rules if re.match(pattern, rule['expectation'], re.IGNORECASE)]

    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(name=suite_name)
    )
        
    split_rules = []

    for rule in rules:
        components = [None] + _get_rule_components(rule['expectation']) # item at odd order = operator, item at even order = value. E.g. [None, 'required', 'and', 'max_length(3)]

        for rule_idx in range(0, len(components), 2): 
            split_rule = rule.copy()
            split_rule['operator'] = components[rule_idx] #Operator OR/AND
            split_rule['expectation_single'] = components[rule_idx+1] #Actual expectation 
            split_rules.append(split_rule)
    
    for rule in split_rules:
        suite.add_expectation(
            create_expectation(rule)
        )

    return suite

# COMMAND ----------

# DBTITLE 1,def create_batch_definition
def create_batch_definition(
    context: gx.data_context.AbstractDataContext,
    data_source_name: str = "spark", 
    data_asset_name: str = "spark dataframe asset", 
    batch_definition_name: str = "batch definition"
) -> gx.core.batch_definition.BatchDefinition:
    """
    Creates a batch definition for a Spark dataframe asset.

    Args:
    - context: The Great Expectation context object.
    - data_source_name (str): The name of the data source.
    - data_asset_name (str): The name of the data asset.
    - batch_definition_name (str): The name of the batch definition.

    Returns:
    - gx.core.batch_definition.BatchDefinition: The batch definition object.
    """
    return context.data_sources\
        .add_spark(name=data_source_name)\
        .add_dataframe_asset(name=data_asset_name)\
        .add_batch_definition_whole_dataframe(name=batch_definition_name)


# COMMAND ----------

# DBTITLE 1,def get_rules
def get_rules(system_id: int, table_name: str) -> list[dict]:
    """
    Retrieves the data quality rules for a specific system and table.

    Args:
    - system_id (int): The ID of the system.
    - table_name (str): The name of the table.

    Returns:
    - list: A list of dictionaries representing the data quality rules.
    """
    df_rules = spark.read.table(METADATA_DATAQUALITY_TABLE)
    
    return df_rules\
        .withColumn('table_name', F.lower(F.col('table_name')))\
        .filter(
            (F.col('table_name') == table_name.lower()) 
            & (F.col('source_system_id') == system_id) 
            & (F.col('active') == 1)
            & (F.col('expectation').isNotNull())
        )\
        .select(
            'table_name',
            'field_name',
            'expectation',
            'expectation_action'
        )\
        .toPandas()\
        .to_dict('records')
        

# COMMAND ----------

# DBTITLE 1,def get_validation
def get_validation(context, batch_definition, suite, table_name, batch_id) -> gx.core.validation_definition.ValidationDefinition:
    """
    Creates a validation definition for a batch definition and expectation suite.

    Args:
    - context: The context object.
    - batch_definition: The batch definition object.
    - suite: The expectation suite object.

    Returns:
    - gx.core.validation_definition.ValidationDefinition: The validation definition object.
    """
    return context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name=f"validation-{table_name}-{batch_id}",
            data=batch_definition,
            suite=suite,
        )
    )

# COMMAND ----------

# DBTITLE 1,def get_checkpoint
def get_checkpoint(context, validation_definition, table_name, batch_id) -> gx.checkpoint.checkpoint.Checkpoint:
    """
    Creates a checkpoint for a validation definition.

    Args:
    - context: The context object.
    - validation_definition: The validation definition object.

    Returns:
    - gx.checkpoint.checkpoint.Checkpoint: The checkpoint object.
    """
    result_format = {
        "result_format": "COMPLETE",
        "unexpected_index_column_names": ["RowIndex"],
    }

    return context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name=f"checkpoint-{table_name}-{batch_id}", 
            validation_definitions=[validation_definition],
            result_format=result_format
        )
    )
    

# COMMAND ----------

# DBTITLE 1,def get_dataframe
def get_dataframe(table_name, batch_id) -> DataFrame:
    """
    Retrieves the dataframe for a specific table and batch ID.

    Args:
    - table_name (str): The name of the table.
    - batch_id (str): The ID of the batch.

    Returns:
    - DataFrame: The dataframe containing the data for the specified table and batch ID.
    """
    path = f'cpdata_catalog_{ENVIRONMENT}.{BRONZE_CONTAINER_NAME}.{table_name}'
    df = spark.read.table(path)
    
    window = Window.orderBy(F.col('RowIndex'))
    
    return df\
        .filter(df.batch_id == batch_id)\
        .withColumn("RowIndex", F.monotonically_increasing_id())\
        .withColumn('RowIndex', F.row_number().over(window))

# COMMAND ----------

# DBTITLE 1,def _combine
def _combine_success(group) -> bool:
    expression = ""
    for row in group.itertuples(index=False):
        if not expression:
            # Start with the first boolean value
            expression += str(row.success)
        else:
            # Append the operator and the boolean value
            expression += f" {row.operator} {row.success}"

    # Evaluate the expression. e.g. eval('True or False and True') <=> True or False and True
    return eval(expression)
    

def _combine_index_list(group) -> list[dict]:
    '''        
        success = true => no return
        success = false => union 
    '''
    success = _combine_success(group)
    if success:
        return []
    
    unique_index_set = {tuple(row.items()): row for sublist in group['unexpected_index_list'] for row in sublist}
    return list(unique_index_set.values())

# COMMAND ----------

# DBTITLE 1,def get_dataquality_df
def get_dataquality_df(checkpoint_result, table_name, batch_id) -> DataFrame:
    """
    Converts the checkpoint result into a DataFrame containing data quality.

    Args:
    - checkpoint_result: The result of running the checkpoint.
    - table_name (str): The name of the table.
    - batch_id (str): The ID of the batch.

    Returns:
    - DataFrame: A DataFrame containing the data quality information.
    """
    expectation_results = list(checkpoint_result.run_results.values())[0]['results']
  
    columns = [
        'table_name',
        'column_name',
        'expectation',
        'success',
        'missing_count',
        'unexpected_count',
        'unexpected_index_list',
        'action',
        'proccesed_datetime',
        'batch_id',
    ]
    
    valid_expectation_results = [result for result in expectation_results if result['result'] or result['success']]

    if not valid_expectation_results:
        raise ValueError('No valid expectation results found, please check the expectation config again')

    df_quality_pandas = pd.json_normalize([dict(item) for item in valid_expectation_results])

    # Remove column name prefix
    df_quality_pandas.columns = df_quality_pandas.columns.str.replace('result.', '', regex=False)

    df_quality_pandas['action'] = df_quality_pandas['expectation_config'].apply(lambda x: x['meta'].get('action'))
    df_quality_pandas['expectation'] = df_quality_pandas['expectation_config'].apply(lambda x: x['meta'].get('expectation'))
    df_quality_pandas['operator'] = df_quality_pandas['expectation_config'].apply(lambda x: x['meta'].get('operator'))
    df_quality_pandas['column_name'] = df_quality_pandas['expectation_config'].apply(lambda x: x['kwargs']['column'])
    df_quality_pandas['missing_count'] = df_quality_pandas['missing_count'].fillna(0).astype(int) if 'missing_count' in df_quality_pandas.columns else 0

    if 'unexpected_index_list' not in df_quality_pandas.columns:
        df_quality_pandas['unexpected_index_list'] = [[] for _ in range(len(df_quality_pandas))]
    else:
        df_quality_pandas['unexpected_index_list']  = df_quality_pandas['unexpected_index_list'].fillna('').apply(list)


    df_quality_pandas = df_quality_pandas.groupby(['column_name', 'expectation', 'action']).apply(lambda group: pd.Series(dict(
        success=_combine_success(group),
        unexpected_index_list=_combine_index_list(group),
        missing_count=group['missing_count'].sum(),
    ))).reset_index()

    
    df_quality_pandas['table_name'] = table_name
    df_quality_pandas['batch_id'] = batch_id
    df_quality_pandas['proccesed_datetime'] = pd.Timestamp.now()
    df_quality_pandas['unexpected_count'] = df_quality_pandas['unexpected_index_list'].apply(lambda index_list: len(index_list))

    return df_quality_pandas
 

# COMMAND ----------

# DBTITLE 1,def check_data_quality
def check_data_quality(
    system_id: int, 
    system_name: str, 
    table_name: str, 
    batch_id: str,
    rules: list[dict] = None,
    df: DataFrame = None,
    **kwargs
) -> tuple[DataFrame, DataFrame]:
    """
    Performs data quality checks on a specified table using a set of test rules.
    
    Args:
    - system_id (int): The ID of the system.
    - system_name (str): The name of the system.
    - table_name (str): The name of the table to perform data quality checks on.
    - batch_id (str): The ID of the batch.
    - rules (list[dict]): A list of expectation rules to apply data quality checks, for example 
        [
            {
                'table_name': 'account', 
                'field_name': 'acctcreateddate', 
                'expectation': 'not_future_date', 
                'expectation_action': 'drop'
            }
        ]
    - df(DataFrame): The spark dataframe to perform data quality checks on.

    - **kwargs: Additional keyword arguments.
    
    Returns:
    - DataFrame: Dataframe that reflects the bronze table with generated RowIndex field 
    - DataFrame: DataFrame containing the data quality results.
    """
    
    # 0. Get Context
    full_table_name = f'{system_name.lower()}_{table_name.lower()}'
    context = gx.get_context()

    # 1. Create Batch
    print('>>>>>>>>>Create Batch Definition...<<<<<<<<<<<')
    batch_definition = create_batch_definition(
        context=context,
        data_source_name=f'spark-{full_table_name}-{batch_id}',
        data_asset_name=batch_id,
        batch_definition_name=f'batch-{full_table_name}-{batch_id}',
    )

    # 2. Get Test Rules / Data Quality Metadata
    print('>>>>>>>>>Get Test Rules...<<<<<<<<<<<')
    rules = rules or get_rules(
            system_id=system_id, 
            table_name=table_name
        )
    print(f'{rules = }')

    # 3. Connect to data source
    print('>>>>>>>>>Connect to Data Source...<<<<<<<<<<<')
    df = df or get_dataframe(
        table_name=full_table_name,
        batch_id=batch_id
    )

    # 4. Create Expectation suite 
    print('>>>>>>>>>Create Expectation suite...<<<<<<<<<<<')
    suite = create_expectation_suite(
        context=context,
        suite_name=f'test-suite-{full_table_name}', 
        rules=rules
    )

    # 5. Create Validation definition
    print('>>>>>>>>>Create Validation Definition...<<<<<<<<<<<')
    validation_definition = get_validation(
        context=context, 
        batch_definition=batch_definition, 
        suite=suite,
        table_name=table_name,
        batch_id=batch_id
    )

    # 6. Create Checkpoint
    print('>>>>>>>>>Define Checkpoint...<<<<<<<<<<<')
    checkpoint = get_checkpoint(
        context=context,
        validation_definition=validation_definition,
        table_name=table_name,
        batch_id=batch_id
    )

    # 7. Run Checkpoint
    print('>>>>>>>>>Run Checkpoint...<<<<<<<<<<<')
    batch_parameters = {"dataframe": df}
    checkpoint_result = checkpoint.run(batch_parameters=batch_parameters)

    # 8. Return Data Quality Dataframe
    return df, get_dataquality_df(checkpoint_result, table_name, batch_id)
