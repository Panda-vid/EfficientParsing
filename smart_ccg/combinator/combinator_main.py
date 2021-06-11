from smart_ccg.combinator.ui.CombinatorView import CombinatorView
from smart_ccg.combinator.util.CombinatorStorage import CombinatorStorage

if __name__ == '__main__':
    dsl_filepath = "resources/combinator/small.dsl"
    wikitablequestions_path = "resources/combinator/WikiTableQuestions"
    storage_path = "resources/combinator/storage"
    storage = CombinatorStorage(storage_path)
    combinator_view = CombinatorView.create_instance(storage, dsl_filepath, wikitablequestions_path)
