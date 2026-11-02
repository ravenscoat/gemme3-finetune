"""Plain-language content for the evidence-backed hiring assessment report."""


def write_sections(page,p,sub,table,example,result_table,subcategory_count,comparison,distribution_tables):
    page('Teaching Gemma to create synthetic data',True)
    p('Hiring assessment report | 2 November 2026')
    sub('What we were asked to do')
    p('Teach Gemma to create new examples that look like the datasets it learned from. This extra training is called fine-tuning. The new examples are called synthetic data.')
    p('The requirement was: "Matches the ranges, distribution, and characteristics of the data it was trained on."')
    table(['Part of the requirement','What we checked'],[
        ['Ranges','Does the model produce a similar spread of values? For example, short and long answers, or younger and older patients.'],
        ['Distribution','Does it produce common topics often and rare topics less often, in similar proportions to the source?'],
        ['Characteristics','Do the examples have the right fields, writing style and patterns? Do their answers and actions fit the situation?']], [36,138])
    sub('What we built')
    p('We trained four separate Gemma 3 4B adapters, one for each dataset, on a laptop GPU with 8 GB of memory. An adapter is a small set of learned changes that works with the original model. We also saved the data splits, model files, generated examples and evaluation scores, and built a local website to explore them.')
    table(['Main achievement','Evidence'],[
        ['Correct record format','All examples in the chosen evaluation batches had the required format: Bitext 156/156, InsuranceQA 128/128, Reliability 64/64 and MIMIC demo 64/64.'],
        ['Closer insurance-data patterns','Adding topic instructions improved the SDMetrics similarity score from 0.7893 to 0.9465 for Bitext, and from 0.6879 to 0.8228 for Reliability.'],
        ['Closer clinical-number patterns','A separate clinical version reduced the age-distribution gap from 7.249 to 2.535 years, and the stay-length gap from 2.003 to 1.040 days.']], [49,125])
    p('These results show that fine-tuning and carefully chosen prompts can bring generated data closer to its source. The following pages explain the data, the training and the evidence in simple terms.')

    page('1. The four datasets')
    p('A dataset is a collection of examples. One row is one example. Its columns, also called fields, hold the pieces of information in that example.')
    table(['Dataset','Prepared examples','What one row contains'],[
        ['Bitext Insurance','39,000','A customer request, a support reply and labels describing the request.'],
        ['InsuranceQA V2','16,813 after overlap removal','An insurance question and one answer marked as suitable in the source.'],
        ['Insurance Reliability','510','A customer situation and the actions an insurance assistant should take.'],
        ['MIMIC-IV demo','275 admissions from 100 patients','Information about one hospital admission. A patient can have more than one admission.']], [38,44,92])
    sub('The main columns and groups')
    table(['Dataset','Fields and groups'],[
        ['Bitext','5 fields: intent (request type), category (broader topic), tags (language style), instruction (customer message) and response (reply). There are 39 intents, 17 categories and 10 possible tag characters.'],
        ['InsuranceQA','3 fields: domain (insurance type), question and answer. There are 12 insurance types. The topics occur at different rates.'],
        ['Reliability','5 main fields: category, subcategory, difficulty, input and expected_output. The last two contain smaller fields. There are 10 categories, '+str(subcategory_count)+' subcategories and 3 difficulty levels.'],
        ['MIMIC demo','13 fields, including age, gender, admission type, insurance, race, hospital stay, outcome, diagnosis and three lab measurements. There are 9 admission types, 3 insurance labels and 14 race labels.']], [36,138])
    sub('Why the datasets behave differently')
    p('Bitext has regular support patterns. InsuranceQA has a wider mix of questions and longer explanations; it was originally designed to help select an answer. Reliability needs several fields to agree with each other. MIMIC is a small clinical dataset with numbers and missing measurements.')
    p('We stored the prepared data as JSONL: one record on each line. This also keeps information inside other fields, such as conversation history. Source IDs help trace a row, but the model is not asked to generate those IDs.')
    p('For MIMIC, we used the official 100-patient demo. Our target is this demo data, rather than the whole U.S. patient population.')

    page('Data quality and preparation')
    p('Before training, we checked the files, kept the fields needed for each task and separated records for learning and evaluation. The aim was to make training consistent and reduce overlap between the splits.')
    sub('Bitext: check the fields and group related examples')
    p('We checked that all five required fields were present and non-empty, and that each intent had a consistent category. The exact-duplicate check found no rows to remove. We grouped records sharing normalized requests, request word sets or replies into the same split. Normalized means ignoring differences such as letter case and punctuation. This reduces overlap, although similar paraphrases can remain.')
    sub('InsuranceQA: decode the text and keep suitable answers')
    p('We decoded the original files into readable questions and answers. For each question, we selected one answer labelled suitable by the source, rather than using the alternative candidate answers. We removed overlapping questions or selected answers from the lower-priority splits: 39 validation rows and 37 test rows. The original answer wording was preserved.')
    sub('Reliability: preserve the scenario and workflow')
    p('We kept the customer message, policy context, conversation history and expected workflow. We checked for repeated inputs across the official splits; none needed removal. Administrative fields such as IDs, source metadata and evaluation criteria were excluded from generation targets.')
    sub('MIMIC: join records and check measurements')
    p('We joined patient, admission, diagnosis and lab tables into one admission record. We used the first listed diagnosis and the earliest finite numeric lab measurement during the hospital stay, with the expected unit. Missing measurements stayed null. We calculated age and stay length and checked that they were not negative. All admissions from one patient stayed in the same split.')
    sub('Checks and records kept across the project')
    table(['Preparation step','Why it matters'],[
        ['Consistent JSONL files and schemas','Each record has a defined set of fields, including nested information where needed.'],
        ['Token-length audits','Check whether records fit the model\'s training length limits.'],
        ['Saved source IDs, seeds and file checksums','Trace examples and reproduce the split. IDs were excluded from what the model was asked to generate.']], [60,114])
    p('These checks cover structure, overlap and preparation rules. They do not replace expert checking of every insurance answer or clinical record. The preparation audits and source files are saved with the project evidence.','small')

    page('Real examples: requests and answers')
    p('These tables show actual cells from the training data. They are examples of what the model learned from, not outputs created by the model. Source wording and values are preserved.')
    example('bitext','Bitext: checking insurance coverage','{{WEBSITE_URL}} is a placeholder in the original support template.')
    example('insuranceqa','InsuranceQA: explaining a deductible')

    page('Real examples: workflows and patient data')
    p('The first table shows selected fields from one Reliability scenario. The second shows all 13 target fields from one real demo admission. The source wording and values are unchanged.')
    example('reliability','Reliability: deciding what to do next','ai_handle means the AI is assigned to handle the request.')
    example('mimic_demo','MIMIC demo: one hospital admission','null means no measurement is available. age_topcoded = false means the age was not recorded as 90+. hospital_expire_flag = 0 means no in-hospital death was recorded.')

    page('Training and generated examples: text')
    p('Read each pair side by side to compare the fields, wording and answer style. These are separate records, not answers to the same question. Bitext replies show the opening and first two steps; the QA answers are complete. These examples do not replace the batch-level evaluation.')
    comparison('bitext','Bitext: insurance coverage')
    comparison('insuranceqa','InsuranceQA: health-insurance deductibles')
    p('The Bitext intent/category were supplied before generation. The InsuranceQA record came from the retained generic generation method. The saved answers have not been certified by an insurance expert.','small')

    page('Training and generated examples: records')
    p('These tables compare selected workflow fields and all 13 clinical fields. Each column is a separate record. Different values are expected; distribution matching is checked across many records.')
    comparison('reliability','Reliability: a deductible inquiry')
    comparison('mimic_demo','MIMIC demo: hospital-admission records')
    p('Reliability category/subcategory/difficulty were supplied as controls. The clinical example is from the separate conditioned candidate, with diagnosis and missing-lab instructions supplied. It is not a verified clinical record.','small')

    page('2. The model and the computer')
    table(['Item','What we used'],[
        ['Model','Gemma 3 4B. It has about four billion parameters: the learned numbers that shape its responses.'],
        ['GPU','NVIDIA GeForce RTX 5060 Laptop GPU with about 8 GB of GPU memory.'],
        ['Software','Windows, Unsloth, PyTorch and Transformers. Project files and caches are on D:.'],
        ['Training method','QLoRA, which reduces the memory needed for fine-tuning. The original adapters use rank 8 and alpha 16.']], [38,136])
    sub('Why we chose QLoRA')
    p('Training needs memory for the model and for the work done during each update. A four-billion-parameter model stored at 16-bit precision needs roughly 8 GB for its weights alone. That leaves little room for training on this GPU.')
    p('LoRA saves memory by training a small adapter instead of changing every model weight. QLoRA adds another saving: it stores much of the original model at 4-bit precision, a smaller number format. This made our local training practical.')
    p('We also used gradient checkpointing. It saves memory by recalculating some intermediate results when needed. This trades some speed for lower memory use.')
    sub('Why four separate adapters?')
    p('The datasets teach different tasks and formats. We started each adapter from the same original model, then trained it on one dataset. This kept the insurance-support, QA, workflow and clinical results separate and easier to compare.')
    sub('Three training terms')
    table(['Term','Simple meaning'],[
        ['Epoch','One pass through all the training examples.'],
        ['Step','One model update, after processing a group of examples.'],
        ['Checkpoint','A saved version of the adapter that can be evaluated or loaded later.']], [38,136])

    page('3. How the data moved through the project')
    p('Prepare the source data -> split it into three groups -> train an adapter -> check progress -> save a chosen version -> generate new examples -> compare them with the source.')
    table(['Dataset','Training','Validation','Test','Why these numbers?'],[
        ['Bitext','31,200','3,900','3,900','About 80% for learning and 10% each for progress checks and testing.'],
        ['InsuranceQA','12,889','1,961','1,963','Kept the original split, then removed repeated questions or chosen answers across splits.'],
        ['Reliability','357','76','77','Kept the official split: about 70%, 15% and 15%.'],
        ['MIMIC demo','230','21','24','Split patients 80/10/10. Different patients have different numbers of admissions.']], [27,23,24,21,79])
    sub('What the three groups are for')
    p('Training data teaches the adapter. Validation data checks progress during training and helps choose which saved version to keep. Test data checks the chosen model after those decisions.')
    p('We used validation checks during training. Some checks used a fixed smaller sample to save time, rather than scoring every validation row each time. For Bitext, the completed run used all 31,200 training examples: 800 per intent.')
    sub('How we kept the checks useful')
    p('InsuranceQA began with 12,889 / 2,000 / 2,000 rows. Removing 39 validation overlaps and 37 test overlaps left the counts above. We kept one suitable source answer per question, rather than training on the incorrect candidate answers.')
    p('All admissions from one MIMIC patient stay in the same group. Otherwise the model could learn from one admission and be tested on another admission from that same person. Bitext also groups related examples to reduce overlap, although similar wording can still remain.')
    sub('Two different kinds of evaluation')
    p('We checked prediction on held-out source records, meaning records not used to update the model. We also asked the model to generate new batches and measured their patterns. The second check directly addresses the synthetic-data requirement.')
    p('The original test sets have already been used. Later prompt trials are reported as improvement experiments, rather than as results on a new untouched test set.')

    page('4. How we checked similarity')
    p('We asked several simple questions. Each measure below checks one part of the requirement; no single score answers everything.')
    table(['Measure','Question it answers','How to read it'],[
        ['Structure validity','Did the model include the required fields and use the right types of values?','A count such as 128/128 means every example had the required format.'],
        ['SDMetrics quality','Do the selected columns have patterns similar to the source, such as topic shares and answer lengths?','Higher means closer for the fields checked. It is not answer accuracy.'],
        ['MMD-squared','When many generated texts are compared with many source texts, how different are their measured text patterns?','Lower means closer. We check requests, replies and the two together.'],
        ['PRDC','Do the examples resemble the source and cover its variety, instead of repeating the same kind?','Four related checks, explained on the next page.'],
        ['Keyword JS distance','Are important words used at similar rates? For example, does a common source word also appear often in generated text?','Lower means closer. We use the source\'s top 100 content words, plus a group for other words.'],
        ['TV distance','Are topic proportions similar? For example, are health-insurance questions too common or too rare?','Lower means closer.'],
        ['Wasserstein distance','Do numeric values have a similar spread, such as patient ages or hospital stays?','Lower means closer. Its units are those of the value checked, such as years.'],
        ['Percentiles','Does generation include low, typical and high values, or short, typical and long answers?','Compares different parts of the spread, rather than only the smallest and largest values.'],
        ['Copying and meaning review','Are the examples new? Do the answers address the questions, and do the actions make sense?','Check reuse separately from relevance, contradictions and correct handling of a situation.']], [34,87,53])
    sub('A useful distinction')
    p('A record can have the correct format and similar text patterns while still giving a wrong answer. We therefore read examples as well as calculating similarity scores.')

    page('4. How to read the scores')
    sub('PRDC: resemblance and variety')
    p('PRDC uses four measures. Together, they help us see whether generation covers different kinds of source examples.')
    table(['Part','What it means'],[
        ['Precision','How often generated examples are near source examples in the measured text patterns. This is resemblance, not factual accuracy.'],
        ['Recall','How much of the source\'s variety the generated collection reaches.'],
        ['Density','How strongly generated examples cluster around source patterns. It can be above 1; higher is not automatically better.'],
        ['Coverage','How much of the source has a close generated example.']], [38,136])
    sub('We used the same measuring tool')
    p('For the text checks, a fixed GPT2-large encoder turns text into lists of numbers. MMD and PRDC compare patterns in those numbers. Using the same encoder is like using the same ruler for each comparison. It does not check whether an insurance fact is true.')
    p('The encoder reads up to 512 tokens, which are small pieces of text. Longer examples are partly cut off for this measurement. We checked requests, replies and their combination.')
    sub('How many generated examples were measured?')
    p('MMD and PRDC used 128 records per main text collection, except Reliability with 64. SDMetrics and keyword checks used the full valid batch listed on each dataset page. We also saved the exact records and compared source samples of the same size.')
    sub('What a score does and does not mean')
    p('An SDMetrics score of 0.9465 does not mean 94.65% of answers are correct. It means the selected data patterns are similar under that measure. SDMetrics checks different columns for different datasets, so those overall scores should not rank the datasets against one another. For InsuranceQA, it checks only domain and text lengths.')
    p('We also used prediction loss: lower loss means the model became better at predicting source-style text. An early Bitext check used MAUVE, another text-similarity measure. The main results below use the saved benchmark, copying checks and content reviews. Assistant reviews are useful checks, but are not expert-verified accuracy scores.')

    page('Prompt engineering: what we changed')
    p('Prompt engineering means changing the instructions given to a model, then checking whether its responses improve. We tested clearer instructions, source passages, a worked example and supplied topic labels. Most of these trials used the same trained adapter, without further training.')
    sub('What the different prompt changes achieved')
    table(['Dataset','What we changed','What we learned'],[
        ['Bitext','Supplied intent and category before generation.','SDMetrics improved from 0.7893 to 0.9465. Topic and text patterns became closer to the source.'],
        ['Reliability','Supplied category, subcategory and difficulty. Also tested extra consistency rules.','Topic guidance improved SDMetrics from 0.6879 to 0.8228. The extra consistency-rule trial was not kept.'],
        ['InsuranceQA','Tried a source passage and then one worked example showing how to answer.','Some responses became clearer, but source support and distribution matching did not improve consistently. We kept the original method.'],
        ['MIMIC demo','Supplied diagnosis and which lab values should be missing.','Conditions guided generation. The final improvement also involved a separately trained candidate, so it was not a prompt-only gain.']], [27,68,79])
    p('For Bitext and Reliability, the successful method started the JSON with topic labels already filled in; Gemma completed the rest. We selected those labels in source-like proportions. That topic mix was a control we supplied, rather than a frequency the model chose by itself.')
    p('The decision was based on the outputs and scores, not simply on making the prompt longer. These were improvement experiments on development batches, not new untouched-test results.','small')

    page('5. Bitext: training and results')
    p('Bitext teaches insurance customer support. Its regular request-and-reply patterns give the model many examples of how to answer similar types of requests.')
    sub('Training: a small check, then the complete pass')
    p('We first ran 200 steps to check that training worked on the laptop. We then continued for 3,700 more steps, reaching 3,900 updates in total. That completed one pass through all 31,200 training records.')
    table(['Check: 500 attempts each','Original Gemma','200-step trial','Complete pass'],[
        ['Correct record format','483/500','499/500','500/500'],
        ['Request types represented, out of 39','2','37','37'],
        ['Topic-share gap (TV; lower is closer)','0.949','0.173','0.150']], [68,35,35,36])
    sub('Next improvement: tell the model the topic')
    p('We added intent and category labels to generation prompts, choosing them in the source proportions. This gives the model a topic to write about. In equal batches of 156 outputs, SDMetrics rose from 0.7893 to 0.9465, MMD fell from 0.01344 to 0.00714, and keyword distance fell from 0.0683 to 0.0329.')
    sub('Results for the chosen 156-output batch')
    result_table('bitext','selected_prefix')
    sub('What this shows')
    p('Fine-tuning improved the format and expanded the range of request types. Topic prompts then brought topic and text patterns closer to the source. All 39 intents appear in the chosen 156-output batch. The supplied topics are a control we added; they do not show that the model learned those proportions by itself.')
    p('Next focus: more original replies. One of the 156 complete records matched training, and 26.92% of replies reused source wording. We would improve variety while keeping the topic mix close to the source.')

    page('6. InsuranceQA: training and results')
    p('InsuranceQA teaches insurance questions and answers. Its 12 topics are annuities, auto, critical illness, disability, health, home, life, long-term care, Medicare, other insurance, renters and retirement plans. The mix is uneven: life insurance has 3,328 training rows, while other insurance has 29.')
    sub('Training and progress')
    p('We trained a separate adapter for one pass through 12,889 records, taking 1,612 updates. Validation checks selected checkpoint 1,600. Prediction loss fell from 3.8888 to 1.7860: the model became better at predicting the source text. Outputs with the correct format increased from 106/128 to 128/128.')
    p('The longest training example used 1,402 tokens and the limit was 1,536. This means the training examples fit, including the longer answers. The chosen generated outputs also finished without hitting their length limit.')
    sub('Results for the chosen 128-output batch')
    result_table('insuranceqa','original_generic')
    sub('What this shows')
    p('The model learned the question-and-answer format and produced a collection with similar domain and length patterns. SDMetrics quality was 0.8908 for those fields. The text and keyword measures provide additional checks of resemblance.')
    p('No complete generated answer or full record exactly matched training after ignoring letter case and punctuation. This is a copying result, not an answer-correctness score. Thirteen questions did match training questions.')
    sub('What we tried next')
    p('Answer consistency needs more work. In a 30-example assistant review, 15 passed, 4 were unsure and 11 had an issue. Further prompt trials did not improve the overall evidence enough, so we kept the original generation method. In a separate passage-following check, the tuned model matched source distributions better, while original Gemma followed supplied passages more closely.')
    p('Next focus: review source answers and train on examples that show how to write a supported answer. The small reviews identify problems; they do not give an overall error rate for the whole dataset.')

    page('7. Reliability: training and results')
    p('Reliability teaches an insurance assistant what to do in a situation: handle it, request information, take an action or pass it to a human. Its 10 categories cover claims, policy questions and changes, payments, documents, handovers, multi-turn conversations, errors and unusual cases. Difficulty is easy, medium or hard.')
    sub('Training and progress')
    p('We trained for three passes through 357 scenarios, taking 135 updates, with validation checks. Original Gemma produced 0/128 records with the full required format. After fine-tuning, it produced 128/128. This made the workflow records usable for the next checks.')
    sub('Next improvement: guide the scenario type')
    p('We supplied category, subcategory and difficulty labels in the prompt. In matched batches of 64, SDMetrics rose from 0.6879 to 0.8228, MMD fell from 0.01943 to 0.01478, and keyword distance fell from 0.2559 to 0.2075. We supplied these labels; the model generated the remaining workflow fields.')
    sub('Results for the chosen 64-output batch')
    result_table('reliability','selected_prefix')
    sub('What this shows and the next focus')
    p('Fine-tuning established the complex record format, and guided prompts improved the measured topic and text patterns. All 64 selected examples also passed the chosen source-label checks.')
    p('An assistant review marked 45 Pass, 15 Unsure and 4 Issue. The next work is specific: answer deductible questions fully, notice a conflicting policy in the context, and avoid repeating a payment already completed in the history. These reviews are not expert-verified accuracy.')
    p('We tried additional prompts and teacher-generated training examples, but kept the better existing method when the checks did not improve. Since the model trained on this benchmark\'s training split, its held-out results measure performance within the same source, rather than on an unrelated external benchmark.')

    page('8. MIMIC demo: training and results')
    p('MIMIC generates structured hospital-admission records rather than conversations. The 13 fields appear in the example table earlier. Missing lab results stay null, meaning missing. Ages recorded as 90 or above use age = 90 and a flag to show this rule.')
    sub('Training and progress')
    p('The original clinical adapter trained for three passes through 230 admissions, taking 87 updates. Validation selected checkpoint 75. Correctly formatted outputs increased from 127/128 for original Gemma to 128/128 after fine-tuning.')
    p('We then trained a separate candidate using 460 training prompt versions over two passes. Its prompts included a diagnosis and which lab values should be missing. In the matched comparison, both versions received these conditions. Their training budgets differed, so the results do not isolate the effect of just one change.')
    sub('Results for the chosen 64-output candidate batch')
    result_table('mimic_demo','selected_candidate')
    table(['Wasserstein distance; lower is closer','Earlier version','Candidate'],[
        ['Age (years)','7.249','2.535'],['Hospital stay (days)','2.003','1.040'],['Glucose (mg/dL)','19.507','17.058']], [76,49,49])
    sub('What this shows')
    p('The candidate brought age, stay-length and lab-value patterns closer to the demo data. SDMetrics rose from 0.6428 to 0.6705. All 64 outputs had the right format; 63 passed the chosen source-value checks. The gaps above compare collections of values, not prediction errors on individual patients. Diagnosis and missing-lab instructions were supplied as controls.')
    p('Next focus: cover more young and older patients. The source age range was 21-90, while the selected generated batch covered 41-86. Some category and relationship patterns also need work. A larger approved clinical source and clinical review would help. These results describe the 100-patient demo, not verified medical correctness.')

    distribution_tables()

    page('9. What the project demonstrates')
    table(['Dataset','Main result against the requirement'],[
        ['Bitext','More request types, consistently correct record format, and closer topic, text and keyword patterns after guided generation.'],
        ['InsuranceQA','Consistent question-and-answer format, lower prediction loss and similar measured topic/length patterns. Supported answers remain the next priority.'],
        ['Reliability','Learned the full workflow format and improved topic/text patterns with guided prompts. Specific workflow errors give a clear next improvement set.'],
        ['MIMIC demo','Generated valid admission records and improved the spread of ages, hospital stays and lab values in the separate candidate.']], [36,138])
    sub('The work behind the results')
    p('The project covers the full process: prepare four datasets, train separate adapters within an 8 GB GPU limit, save versions, generate examples, calculate comparisons and build a usable demonstration. Saved prompts, source records, outputs and scores let a reviewer check the results.')
    sub('What we learned')
    p('Training helped the model learn each record format. Topic instructions improved some distributions. Additional experiments were kept only when the evidence supported them. This gave us a practical way to improve the data without assuming that more training or a longer prompt would always help.')
    p('Next steps would target supported QA answers, workflow consistency, reply originality and rare clinical groups. A larger model or knowledge distillation, where a stronger model supplies training examples, could then be tested using the same checks.')
    sub('Where to check the evidence')
    table(['Evidence','Project folder or file'],[
        ['Prepared data and prompts','data/<dataset>/prepared/'],
        ['Training progress and initial results','reports/training_stage_comparison.md; reports/all_datasets_assessment.md'],
        ['Benchmark scores, charts and definitions','reports/distribution_benchmark/v1/'],
        ['Guided-prompt comparisons','outputs/prefix-alignment-v1/; outputs/fidelity-alignment-v1/'],
        ['Clinical candidate','outputs/mimic-demo-fidelity-conditioned-v1-windows/'],
        ['QA checks and source review','outputs/insuranceqa-base-diagnostic-v1/; outputs/insuranceqa-one-shot-v1/'],
        ['Reliability prompt / teacher trials','outputs/reliability-consistency-v1/; outputs/reliability-distillation-v2/'],
        ['Decisions and reasons','decision.md']], [61,113])
