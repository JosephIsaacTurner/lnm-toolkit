

I have some data in "/Users/jiturner/Repositories/lnm-toolkit/example_data"
There's a CSV there called "participants.csv" with columns:
subject,sex,age_at_stroke,race,wab_days,wab_aq,wab_type,roi_2mm,t
The 't' column is the nifti path roi networks, and roi_2mm is the path to roi masks (nifti).
We can filter for only rows in the CSV where wab_type is in ['Broca' or 'NoAphasia']
and then run a case control analysis comparing those two groups, controlling for
roi volume and roi centrality.
I want you to write some tests for the functions in analysis.py using this dataset. We will use pytest for testing.
Will we need to mock the file I/O operations in io.py to avoid actually writing files during testing? Or should we allow it to write temporary files and clean them up after the tests?
You can decide the most efficient approach here. 
We'd also want to write some documentation of some kind as a quick start demo using this included dataset, showing people how to run the analysis.