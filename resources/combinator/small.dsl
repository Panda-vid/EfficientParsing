"""
    This file denotes the underlying DSL grammar/operations for the recombinator.
    As one can see the file is divided in subsections we call category.

    Right now there are 5 categories:
    'complete' denoting complete actions of the DSL,
    'sub' denoting subactions of complete actions
    'conjunction' denoting conjunctions (note that these conjunctions also work in conditions)
    'condition' denoting a condition
    'conditionals' boolean function such as ==, >=, <= ...

    Each category is started by a tag of form <category_name> and finished with a tag <category_name>.
    The parser cannot parse nested categories.
    Possible inputs are: column, cell, table, value and condition or an ALLCAPS action.
    For condition bodies the special parameter condition_body is used.

    Each input is surrounded by either [] brackets or () brackets,
    where [] brackets denote that the input has no brackets around it
    and () brackets denote that the input is surrounded by round brackets.
    A comma right after the leading bracket indicates that the input is given as a comma-separated list.
"""

<complete>
    SELECT [,column] [FROM]
    INSERT INTO [table] (,column) (,value)
    DELETE [FROM]
    UPDATE [table] [SET]
<complete>
<sub>
    FROM [table]
    FROM [table] [condition]
    SET [,column=value]
<sub>
<conjunction>
    AND
    OR
<conjunction>
<condition>
    WHERE
<condition>
<conditionals>
    >
    <
    >=
    <=
    ==
    !=
<conditionals>