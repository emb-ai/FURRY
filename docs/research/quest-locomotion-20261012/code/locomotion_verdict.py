"""Prospectively specified diagnostic gait gate; never controls PPO or deployment."""

def summarize(comparison):
    pairs = comparison['paired_gait']
    totals = {}
    for label in ('baseline','candidate'):
        rows = [p[label] for p in pairs]
        seconds = sum(r['seconds'] for r in rows)
        path = sum(r['moving_reference_path_m'] for r in rows)
        totals[label] = {'trips_per_s':sum(r['trips'] for r in rows)/seconds,
                         'braking_trips_per_s':sum(r['braking_trips'] for r in rows)/seconds,
                         'directed_progress_ratio':sum(r['directed_progress_m'] for r in rows)/path if path>.1 else None,
                         'seconds':seconds}
        for metric in ('heading_delta_error_mean_deg','lateral_error_abs_mean_m'):
            totals[label][metric]=sum(r[metric]*r['seconds'] for r in rows)/seconds
        totals[label].update(comparison['matched_prefix_means'][label])
    b,c=totals['baseline'],totals['candidate']
    reasons=[]
    if comparison['new_falls']:reasons.append('new_falls')
    for k,tolerance in [('trips_per_s',.05),('braking_trips_per_s',.05),('slide_rate',.02),('window_displacement_error_m',.05),('heading_delta_error_mean_deg',.05),('lateral_error_abs_mean_m',.05),('upper_joint_rmse',.05),('upper_key_rmse_m',.05)]:
        if k not in b or k not in c:
            reasons.append('missing_'+k);continue
        if c[k]>b[k]*(1+tolerance)+1e-6:reasons.append(k+'_regression')
    if b['directed_progress_ratio'] is not None and c['directed_progress_ratio'] is not None and c['directed_progress_ratio']<b['directed_progress_ratio']-abs(b['directed_progress_ratio'])*.05-1e-6:
        reasons.append('directed_progress_regression')
    reasons += [r for r in comparison['reasons'] if r in ('more_falls','less_survival')]
    improved=any(c[k]<b[k]*.95-1e-6 for k in ('trips_per_s','braking_trips_per_s','slide_rate','window_displacement_error_m') if k in b and k in c)
    return {'gait_non_regressing':not reasons,'gait_accepted':not reasons and improved,'reasons':reasons,'totals':totals,'legacy_accepted':comparison['accepted'],'purpose':'diagnostic, no early stop or automatic deployment','holdout':'previously reused Quest validation, not independent test','thresholds_fixed_before_launch':True}
