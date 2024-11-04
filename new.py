def set_state_profile(prefix, df, unique_fields, state_profile, escheat_date_field_name, escheat_due_date_field_name, state_profiles_prefix):
    """
    Sets the state profile for each record in the dataframe based on various conditions related to the escheat process.

    Parameters:
    - prefix (str): Prefix used for naming dynamically generated columns.
    - df (DataFrame): The input dataframe containing the records to process.
    - unique_fields (list): List of fields used to uniquely identify each record.
    - state_profile (str): The name of the column to store the final state profile ID.
    - escheat_date_field_name (str): The name of the column containing the escheat date.
    - escheat_due_date_field_name (str): The name of the column containing the escheat due date.
    - state_profiles_prefix (str): Prefix used for state profile related columns.

    Returns:
    DataFrame: A dataframe with the state profile set for each record.
    """
    state_profile_df = set_escheat_date(state_profiles_prefix, df, escheat_date_field_name)
    state_profile_df = get_escheat_due_date(state_profiles_prefix, state_profile_df, escheat_date_field_name, escheat_due_date_field_name)

    # create 2 window spec partitions
    window_spec = Window.partitionBy(unique_fields)
    state_partition = (
        window_spec
        .orderBy(
            col(f'{state_profiles_prefix}EffectiveDate'),
            col(f'{state_profiles_prefix}ProfileId')
        )
    )
    reversed_state_partition = (
        window_spec
        .orderBy(
            col(f'{prefix}_ValidProfile').desc(),
            col(f'{state_profiles_prefix}EffectiveDate').desc(),
            col(f'{state_profiles_prefix}ProfileId').desc()
        )
    )

    is_sold_on_or_after_fn = lambda: col(f'{state_profiles_prefix}EffectiveDateType') == 'SOLDONORAFTER'

    # set valid state profile for SOLDONORAFTER (round 1)
    state_profile_df = (
        state_profile_df
        .select(
            '*',
            first(col(escheat_date_field_name), ignorenulls=True).over(state_partition).alias('eDateToUse'),
            when(
                is_sold_on_or_after_fn(),
                when(
                    (col(f'{state_profiles_prefix}EffectiveDate') <= col('SoldDate'))
                    |
                    (col('ReplacementDate').isNotNull() & (col(f'{state_profiles_prefix}EffectiveDate') <= col('ReplacementDate'))),
                    1
                )
                .otherwise(0)
            )
            .otherwise(None)
            .alias(f'{prefix}_ValidProfile_R1')
        )
    )

    # only consider items which below the last ValidProfile_R1 = 1
    state_profile_df = (
        state_profile_df
        .withColumn(
            "next_valid_profile",
            lead(
                when(col(f'{prefix}_ValidProfile_R1') == 1, col(f'{prefix}_ValidProfile_R1')),
                1
            ).over(state_partition)
        )
        .withColumn(
            f'{prefix}_ValidProfile_R1',
            when(
                col("next_valid_profile").isNotNull(),
                0
            )
            .otherwise(col(f'{prefix}_ValidProfile_R1'))
        )
        .drop("next_valid_profile")
    )

    print('1')
    display(state_profile_df)

    # remove all state profile which is not sold on or after but the effective date > escheat date
    state_profile_df = (
        state_profile_df
        .where(col(f'{prefix}_ValidProfile_R1').isNull() | (col(f'{prefix}_ValidProfile_R1') == 1))
        .select(
            '*',
            row_number().over(state_partition).alias('RN')
        )
        .where(
            (col(f'{prefix}_ValidProfile_R1') == 1)
            |
            (
                (col('RN') == 1)
                |
                (col(escheat_date_field_name).isNull() | (col(f'{state_profiles_prefix}EffectiveDate') <= col(escheat_date_field_name)))
            )
        )
    )
    
    # set valid state profile for other effective date types (round 2)
    state_profile_df = (
        state_profile_df
        .withColumn(
            "prev_escheat_date",
            lag(col(escheat_date_field_name), 1).over(state_partition)
        )
        .select(
            "*",
            # ignore all valid state profile in the above statement
            when(col(f'{prefix}_ValidProfile_R1').isNotNull(), col(f'{prefix}_ValidProfile_R1'))

            # if there is no valid item, the valid item is the 1st one
            .when(col('RN') == 1, 1)

            # skip item, if e-date of previous item and e-date of profile is null
            .when(
                coalesce(
                    col("prev_escheat_date"),
                    col('eDateToUse')
                ).isNull(),
                lit(None)
            )

            # the valid item must be less than or equal to the previous item
            .when(
                col(f'{state_profiles_prefix}EffectiveDate') <= coalesce(
                    col("prev_escheat_date"),
                    col(escheat_date_field_name),
                    col('eDateToUse')
                ),
                1
            )
            .otherwise(0)
            .alias(f'{prefix}_ValidProfile')
        )
        .drop("prev_escheat_date")
    )
    
    print('2')
    display(state_profile_df)

    # if there are any profile invalid, all next profile will be invalid
    state_profile_df = (
        state_profile_df
        .withColumn(
            "prev_invalid_profile",
            lag(
                when(col(f'{prefix}_ValidProfile') == 0, True),
                1
            ).over(state_partition)
        )
        .withColumn(
            f'{prefix}_ValidProfile',
            when(
                col("prev_invalid_profile").isNotNull(),
                0
            )
            .otherwise(col(f'{prefix}_ValidProfile'))
        )
        .drop("prev_invalid_profile")
    )
    
    print('3')
    display(state_profile_df)

    # only get the last valid
    rank_df = (
        state_profile_df
        .select(
            '*',
            col(f'{state_profiles_prefix}ID').alias(state_profile),
            row_number().over(reversed_state_partition).alias(f'{prefix}_RankProfile')
        )
        .filter(col(f'{prefix}_RankProfile') == 1)
        .drop('RN', 'eDateToUse', f'{prefix}_ValidProfile_R1', f'{state_profiles_prefix}ProfileId', f'{prefix}_RankProfile')
    )

    return rank_df.dropDuplicates(unique_fields)