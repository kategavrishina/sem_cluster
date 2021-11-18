This file describes the data set WordSenseRus available at https://research.yandex.com/datasets/toloka.

The data set is intended for non-commercial use. You must indicate that the data was obtained using Yandex.Toloka. If you plan to use the datasets for commercial purposes, obtain consent from Yandex by contacting: toloka@support.yandex.com. 

This dataset contains human-annotated sense identifiers for 2562 contexts of 20 words used in the RUSSE'2018 shared task on Word Sense Induction and Disambiguation for the Russian language. Assembled by Dmitry Ustalov in 2017. In particular, 80 pre-annotated contexts were used for training the human annotators, and 2562 contexts were annotated by humans such that each context was annotated by 9 different annotators. After the annotation, every context was additionally inspected (“curated”) by the organizers of the shared task.

Accuracy results for some aggregation models are:
Percentage Agreement: 0.906591
Krippendorff Alpha Agreement: 0.825366
Randolph Kappa Agreement: 0.875454

Main tasks are in the file tasks-test.tsv, the format is:
	<INPUT:id>\t<INPUT:lemma>\t<INPUT:left>\t<INPUT:word>\t<INPUT:right>\t<INPUT:senses>
Supplementary file with golden answers and hints is in the file tasks-eval.tsv.xz, the format is:
	<INPUT:id>\t<INPUT:lemma>\t<INPUT:left>\t<INPUT:word>\t<INPUT:right>\t<GOLDEN:sense_id>\t<HINT:text>\t<INPUT:senses>
Training tasks are in the file tasks-train.tsv, the format is:
	<INPUT:id>\t<INPUT:lemma>\t<INPUT:left>\t<INPUT:word>\t<INPUT:right>\t<GOLDEN:sense_id>\t<HINT:text>\t<INPUT:senses>
Full results are in the file assignments_01-12-2017.tsv.xz, the format is:
	<INPUT:id>\t<INPUT:left>\t<INPUT:word>\t<INPUT:lemma>\t<INPUT:right>\t<INPUT:senses>\t<OUTPUT:activity>\t<OUTPUT:sense_id>\t<GOLDEN:activity>\t<GOLDEN:sense_id>\t<HINT:text>\t<ASSIGNMENT:link>\t<ASSIGNMENT:assignment_id>\t<ASSIGNMENT:worker_id>\t<ASSIGNMENT:status>\t<ASSIGNMENT:started>
Curated report is in the file report-curated.tsv.xz, the format is:
	<id>\t<lemma>\t<changed>\t<unsure>\t<prior_sense_id>\t<sense_id>\t<curated_sense_id>\t<confidence>\t<left>\t<word>\t<right>\t<senses>
Aggregated results are in the file aggregated_results_pool_1036853__2017_12_01.tsv, the format is:
	<INPUT:id>\t<INPUT:left>\t<INPUT:lemma>\t<INPUT:right>\t<INPUT:senses>\t<INPUT:word>\t<OUTPUT:sense_id>\t<CONFIDENCE:sense_id>\t<OUTPUT:activity>\t<GOLDEN:sense_id>
Final aggregated dataset is in the file bts-rnc-crowd.tsv, the format is:
	<id>\t<lemma>\t<sense_id>\t<left>\t<word>\t<right>\t<senses>
Annotator agreement report is in the file agreement.txt.

Some statistics about the dataset are below.
	The number of contexts: 2562. 
	The number of training tasks: 79.
	The number of golden tasks: 1132
	The number of aggregated assignments: 10600.