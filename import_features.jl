import Pkg
Pkg.activate("../../../SuperFit.jl")
Pkg.instantiate()
using Turing, MCMCChains
using StatsBase
using CSV, DataFrames
using SuperFit

function import_fitting_params(filename)
    """
    Returns the median fitting parameters across iterations and chains. 
    Only considers the last 1000 iterations of each chain to skip the burnout period.
    """
    trace = MCMCChains.read(filename, Chains)
    A_green, beta_green, gamma_green, t_0_green, tau_rise_green, tau_fall_green, extra_sigma_green, A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = median(Array(trace[end-1000:end,:,:]), dims=1)
    return (;A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma, A_green, beta_green, gamma_green, t_0_green, tau_rise_green, tau_fall_green, extra_sigma_green)
    
end

function generate_simplified_label_csv(label_csv, new_csv)
    """
    Converts label CSV into DataFrame with two columns:
    "ZTF_name" and "type". Then only keeps rows with allowed
    types.
    """
    println("Importing CSV")
    df = DataFrame(CSV.File(label_csv; header=1))
    
    println("Reducing to two columns")
    df_subset = df[:, filter(x -> (x in ["type","ZTF_name"]), names(df))]
    
    allowed_types = ["SN Ia", "SN II", "SN 1b", "SN 1c", "SN 1b/c", "SN IIn", "SLSN-I", "SLSN-II"]
    println("Filtering dataset to allowed supernova types")
    
    df_filtered = filter(row -> (!ismissing(row.type) && row.type in allowed_types), df_subset)
    CSV.write(new_csv, df_filtered)
end

function import_label_csv(label_csv)
    """
    Converts label CSV into DataFrame with two columns:
    "ZTF_name" and "type"
    """
    println("Importing CSV")
    df = DataFrame(CSV.File(label_csv; header=1))
    return df
end

        
function import_class_label(filename, label_dataframe)
    """
    Returns the supernova type associated with each fit file.
    Assumes that the filename is the ZTF supernova name.
    """
    ztf_name = chop(filename, tail=4)
    return filter(row -> row.ZTF_name == ztf_name, label_dataframe).type[1]
end


function generate_input_csv(fits_dir, label_csv, save_csv)
    
    label_df = import_label_csv(label_csv)

    all_ztf_files = readdir(fits_dir)
    ct = 0
    #input_named_tuple = NamedTuple(name=Vector{String}(), label=Vector{String}())
    for fn in all_ztf_files
        ct += 1
        try
            println(fn)
            fit_params = import_fitting_params(fits_dir * fn)
            field_names = keys(fit_params)
            input_names = NamedTuple{(:name, :class, field_names...)}
            tuple_vals = [[], []]
            for k in field_names
                push!(tuple_vals, [])
            end
            input_named_tuple = input_names(tuple_vals)
            label = import_class_label(fn, label_df)
            ztf_name = chop(fn, tail=4)
            all_params = (; ztf_name, label, fit_params...)
            for (fit_val, val_arr) in zip(all_params, input_named_tuple)
                push!(val_arr, fit_val)
            end
            if ct == 1
                CSV.write(save_csv, input_named_tuple)
            else       
                CSV.write(save_csv, input_named_tuple, append=true)
            end
        catch e
            println("Import failed")
            continue
        end
    end
end

fits_dir = "../../../ZTF_fits_multiband/"
label_csv = "filtered_labels.csv"
save_csv = "input_data.csv"
generate_input_csv(fits_dir, label_csv, save_csv)

        
    
    
    
    