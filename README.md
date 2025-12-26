# MARL_signal_SV_schedule

This is the demo code for our paper "**_Schedule Adherence Oriented Traffic Signal Control for Large-Scale Scheduled Vehicle Operations: A Multi-Agent Reinforcement Learning Approach_**":

## **Abstract**
>Travel time reliability plays a crucial role in modern transportation systems, particularly for Scheduled Vehicles (SVs) that operate according to predefined timetables. SVs are becoming increasingly prevalent in urban transportation systems, with common examples including buses, delivery trucks, ride-hailing service vehicles, and travel reservation vehicles. As the timetable
of SVs can be seen as a form of promise, it is critical to ensure their schedule adherence by enhancing the travel time reliability through urban traffic management and control strategies.
Among various methods in the literature, signal control appears to be a promising strategy to enhance the schedule adherence of SVs. Nevertheless, existing approaches remain limited in
addressing the challenges posed by large-scale SV operations. This study addresses the problem of large-scale SV signal coordination by proposing a Multi-Agent Reinforcement Learning
(MARL) framework that ensures schedule adherence while maintaining overall traffic efficiency. We propose a SV priority allocation mechanism that incorporates a distance-based weight,
which gives higher priority to SVs closer to their destinations. It enables signal control agents to precisely identify and prioritize SVs with different schedules and conflicting directions. To
effectively capture each SV’s observed information, we design a Transformer-based feature extractor to process the variable-length observation of SV. The proposed Transformer-Enhanced
MARL method addresses the trade-off between individual SV schedule adherence and overall traffic efficiency through a comprehensive observation and a well-structured reward. Simulations
in both a grid network and a real-world network indicate that the proposed method can significantly improve schedule adherence for SV while simultaneously reducing overall traffic
congestion compared to benchmarks. Sensitivity analysis further confirms the adaptability of the proposed method across varying traffic conditions. Results demonstrate that our proposed
method can ensure schedule adherence for large-scale SVs without sacrificing significant system efficiency.

![fig1](https://github.com/user-attachments/assets/5830c564-6888-4267-9e0b-d17ee1f1f343)

## **Run**
Train and test a new model of our proposed method:

`python main_transformer.py`

Model evaluation:

`main_evaluate.py`
