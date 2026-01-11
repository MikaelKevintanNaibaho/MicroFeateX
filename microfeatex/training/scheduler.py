class HyperParamScheduler:
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def get_value(self, step, start_val, end_val, start_step_pct=0.0, end_step_pct=1.0):
        """
        Generic linear scheduler.
        """
        curr_pct = step / self.total_steps

        # Before start phase
        if curr_pct < start_step_pct:
            return start_val

        # After end phase
        if curr_pct > end_step_pct:
            return end_val

        # Linear Interpolation
        phase_pct = (curr_pct - start_step_pct) / (end_step_pct - start_step_pct)
        return start_val + (end_val - start_val) * phase_pct
