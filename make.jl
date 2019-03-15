using Distributed
cd(@__DIR__)
addprocs(4)
@everywhere using Weave
@everywhere cd(@__DIR__)

##
files = [
"reinforce.jmd",
"cem.jmd",
# "ql.jmd",
]

for file in files
    @spawn weave(file, doctype="md2html")
    @spawn convert_doc(file, file[1:end-3]*"ipynb")
end
