#!/usr/bin/env python

import json

import argparse

from os.path import exists

# python dada_header_code_generator.py -j dada_header.json -H ../include/dada_header.h -s ../src/dada_header.c

parser = argparse.ArgumentParser(description='Generate C code to get/set PSRDADA ascii header.')
parser.add_argument('-j', '--json_file_name')
parser.add_argument('-H', '--header_file_name')
parser.add_argument('-s', '--source_file_name')

args = parser.parse_args()
json_file_name = args.json_file_name
header_file_name = args.header_file_name
source_file_name = args.source_file_name

# Create file handles for data read and write
json_file = open(json_file_name, "r")
dada_header = json.load(json_file)
json_file.close()

# First we need a header file
header_file = open(header_file_name, "w")

header_file.write("#ifndef __DADA_HEADER_H\n")
header_file.write("#define __DADA_HEADER_H\n\n")

header_file.write("#define DADA_STRLEN 1024\n\n")
header_file.write("#ifdef __cplusplus\n")
header_file.write("extern \"C\" {\n")
header_file.write("#endif\n\n")

header_file.write("#include \"inttypes.h\"\n\n")

header_file.write("  typedef struct dada_header_t{\n")
for key in dada_header:
    data_type = dada_header[key]
    print(key, data_type)
    if data_type == "string":
        header_file.write(f"    char {key.lower()}[DADA_STRLEN];\n")
    else:
        header_file.write(f"    {data_type} {key.lower()};\n")

header_file.write("  }dada_header_t;\n\n")

header_file.write("  int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header);\n\n")
header_file.write("  int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer);\n\n")

header_file.write("  int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header);\n\n")
header_file.write("  int write_dada_header_to_file(const dada_header_t dada_header,const char *dada_header_file_name);\n\n")

header_file.write("#ifdef __cplusplus\n")
header_file.write("}\n")
header_file.write("#endif\n\n")

header_file.write("#endif\n")
header_file.close()

# Now we write source code
source_file = open(source_file_name, "w")

source_file.write("#ifndef _GNU_SOURCE\n")
source_file.write("#define _GNU_SOURCE\n")
source_file.write("#endif\n\n")

source_file.write("#include \"futils.h\"\n")
source_file.write("#include \"dada_def.h\"\n")
source_file.write("#include \"ascii_header.h\"\n\n")
source_file.write(f"#include \"{header_file_name}\"\n\n")

source_file.write("#include <stdlib.h>\n")
source_file.write("#include <stdio.h>\n\n")
source_file.write("#include <string.h>\n\n")

# read from file function

source_file.write("int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header){\n\n")
source_file.write("  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);\n")
source_file.write("  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);\n\n")
source_file.write("  fileread(dada_header_file_name, dada_header_buffer, DADA_DEFAULT_HEADER_SIZE);\n")
source_file.write("  read_dada_header(dada_header_buffer, dada_header);\n\n")
source_file.write("  free(dada_header_buffer);\n\n")
source_file.write("  return EXIT_SUCCESS;\n}\n\n")

# read function
source_file.write("int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header){\n\n")

for key in dada_header:
    data_type = dada_header[key]
    print(key, data_type)

    #print(f'  if \(ascii_header_get\(dada_header_buffer, "{key}", "%d", dada_header.{key.lower\(\)}\) < 0\)  {\n')
    if data_type == "int":
        data_type_marker = "\"%d\""
    if data_type == "float":
        data_type_marker = "\"%f\""
    if data_type == "double":
        data_type_marker = "\"%lf\""
    if data_type == "string":
        data_type_marker = "\"%s\""
    if data_type == "uint64_t":
        data_type_marker = "\"%\" PRIu64 \"\""

    source_file.write(f"  if (ascii_header_get(dada_header_buffer, \"{key}\", {data_type_marker}, &dada_header->{key.lower()}) < 0)")
    source_file.write("  {\n")
    source_file.write(f"    fprintf(stderr, \"WRITE_DADA_HEADER_ERROR: Error getting {key}, \"\n")
    source_file.write("            \"which happens at %s, line [%d].\\n\",\n",)
    source_file.write("            __FILE__, __LINE__);\n")
    source_file.write("    exit(EXIT_FAILURE);\n")
    source_file.write("  }\n\n")

source_file.write("  return EXIT_SUCCESS;\n}\n\n")

# write to file function
source_file.write("int write_dada_header_to_file(const dada_header_t dada_header, const char *dada_header_file_name){\n\n")
source_file.write("  FILE *fp = fopen(dada_header_file_name, \"w\");\n")
source_file.write("  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);\n")
source_file.write("  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);\n\n")
source_file.write("  sprintf(dada_header_buffer, \"HDR_VERSION  1.0\\nHDR_SIZE     4096\\n\");\n")
source_file.write("  write_dada_header(dada_header, dada_header_buffer);\n")  
source_file.write("  fprintf(fp, \"%s\\n\", dada_header_buffer);\n\n")
source_file.write("  free(dada_header_buffer);\n")
source_file.write("  fclose(fp);\n\n")
source_file.write("  return EXIT_SUCCESS;\n}\n\n")

# write function
source_file.write("int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer){\n\n")

for key in dada_header:
    data_type = dada_header[key]
    print(key, data_type)

    #print(f'  if \(ascii_header_get\(dada_header_buffer, "{key}", "%d", dada_header.{key.lower\(\)}\) < 0\)  {\n')
    if data_type == "int":
        data_type_marker = "\"%d\""
    if data_type == "float" or data_type == "double" :
        data_type_marker = "\"%f\""
    if data_type == "string":
        data_type_marker = "\"%s\""
    if data_type == "uint64_t":
        data_type_marker = "\"%\" PRIu64 \"\""
        
    source_file.write(f"  if (ascii_header_set(dada_header_buffer, \"{key}\", {data_type_marker}, dada_header.{key.lower()}) < 0)")
    source_file.write("  {\n")
    source_file.write(f"    fprintf(stderr, \"READ_DADA_HEADER_ERROR: Error setting {key}, \"\n")
    source_file.write("            \"which happens at %s, line [%d].\\n\",\n",)
    source_file.write("            __FILE__, __LINE__);\n")
    source_file.write("    exit(EXIT_FAILURE);\n")
    source_file.write("  }\n\n")

source_file.write("  return EXIT_SUCCESS;\n}\n")

source_file.close()
