"""Implement custom exceptions and error handling for processing protein data."""

import os

import sidechainnet

ERROR_CODES = [
    ("SEQUENCE_ERRORS",
     "structures failed because of an unresolvable issue with sequences."),
    ("MULTIPLE_CONTIG_ERRORS",
     "structures failed to parse because they had ambiguous sequence "
     "alignments/multiple identical contigs."),
    ("FAILED_ASTRAL_IDS", "Astral IDs failed to parse for some unknown reason."),
    ("TEST_PARSING_ERRORS", "test structures experience AttributeErrors when parsing."),
    ("NSAA_ERRORS", "structures failed because they had non-standard amino acids."),
    ("MISSING_ASTRAL_IDS", "Astral IDs could not be found in the lookup file."),
    ("SHORT_ERRORS", "structures failed because they were too short."),
    ("PARSING_ERROR_ATTRIBUTE",
     "structures raised AttributeErrors when parsing PDB/mmCIF files."),
    ("PARSING_ERROR",
     "structures raised `pr.proteins.pdbfile.PDBParseError` when parsing PDB/mmCIF"
     " files."),
    ("PARSING_ERROR_OSERROR",
     "structures experienced OSErrors when parsing PDB/mmCIF files."),
    ("UNKNOWN_EXCEPTIONS",
     "structures experienced Unknown exceptions when parsing PDB/mmCIF files."),
    ("MISSING_ATOMS_ERROR", "structures failed because they had missing atoms."),
    ("NONE_STRUCTURE_ERRORS", "structures were returned as None from subroutine."),
    ("NONE_CHAINS", "chains became none when parsing."),
    ("COORDSET_INDEX_ERROR", "structures failed to correctly select ACSIndex."),
    ("D_AMINO_ACIDS", "structures contain d amino acids.")
]


class ProteinErrors(object):
    """A simple, flexible class to record and report errors when parsing.

    For each type of error, this class records the error's name (a short, descriptive
    title), the error's description, and an integer error code.

    The reason for implementing this extra class is that when using multiprocessing to
    process all the structures, the processes are free to fail by raising any type of
    special exception. To communicate this information back to the parent process, we
    create a simple mapping from error to code/description so that these can be shown
    to the user.
    """

    def __init__(self):
        self.code_to_name = {}
        self.code_to_descr = {}
        self.name_to_code = {}
        self.name_to_descr = {}
        self.counts = None
        for i, (error_name, error_descr) in enumerate(ERROR_CODES):
            self.name_to_code[error_name] = i
            self.name_to_descr[error_name] = error_descr
            self.code_to_name[i] = error_name
            self.code_to_descr[i] = error_descr
        self.error_codes_inv = None

    def __getitem__(self, error_name):
        """Return the error code for a certain error name."""
        return self.name_to_code[error_name]

    def count(self, ec, pnid):
        """Create a record of a certain PNID exhibiting a certain error."""
        if not self.counts:
            self.counts = {ec: [] for ec in self.name_to_code.values()}

        self.counts[ec].append(pnid)

    def summarize(self, total_processed=None):
        """Print a summary of all errors that have been recorded."""
        if not self.counts:
            print("No errors recorded.")
            return

        print("The following errors occurred:")
        self.error_codes_inv = {v: k for k, v in self.name_to_code.items()}
        for error_code, count_list in self.counts.items():
            if len(count_list) > 0:
                name = self.error_codes_inv[error_code]
                descr = self.name_to_descr[name]
                if total_processed:
                    percent = f"{'(' + str(int(len(count_list) / total_processed * 100)) + '%)'}"
                    print(
                        f"{name + ':':<25}{str(len(count_list)):^8} {percent:^6} {descr}")
                else:
                    print(f"{name + ':':<25}{len(count_list):^8}{descr}")
        print("")
        self.write_summary_files()

    def get_pnids_with_error_name(self, error_name):
        """After counting, returns a list of pnids that have failed with a specified error
        code."""
        error_code = self[error_name]
        return self.counts[error_code]

    def get_error_names(self):
        """Returns a list of error names."""
        return self.name_to_code.keys()

    def write_summary_files(self):
        """For all counted errors, writes the list of pnids with each error to the errors/
        directory."""
        os.makedirs("errors/", exist_ok=True)
        for e in self.get_error_names():
            if len(self.get_pnids_with_error_name(e)) > 0:
                with open(f"errors/{e}.txt", "w") as f:
                    f.write("\n".join(self.get_pnids_with_error_name(e)) + "\n")

    def get_error_name_from_code(self, code):
        """Returns the error name for the associated code."""
        return self.code_to_name[code]


ERRORS = ProteinErrors()


class IncompleteStructureError(Exception):
    """An exception to raise when a structure is incomplete."""


class NonStandardAminoAcidError(Exception):
    """An exception to raise when a structure contains a Non-standard amino acid."""


class SequenceError(Exception):
    """An exception to raise when a sequence is not as expected."""


class ContigMultipleMatchingError(Exception):
    """An exception to raise when a sequence is ambiguous due to repetitive contigs."""


class ShortStructureError(Exception):
    """An exception to raise when a sequence too short to be meaningful."""


class MissingAtomsError(Exception):
    """An exception to raise when a residue is missing atoms."""


class NoneStructureError(Exception):
    """An exception to raise when a parsed structure becomes None."""


def report_errors(pnids_errorcodes, total_pnids):
    """Provides a summary of errors after parsing SidechainNet data.

    Args:
        pnids_errorcodes: A list of tuples (pnid, error_code) of ProteinNet IDs
            that could not be parsed completely.
        total_pnids: Total number of pnids processed.

    Returns:
        None. Prints summary to stdout and generates files containing failed
        IDs in .errors/{ERROR_CODE}.txt
    """
    print(f"\n{total_pnids} ProteinNet IDs were processed to extract sidechain " f"data.")
    error_summarizer = sidechainnet.utils.errors.ProteinErrors()
    for pnid, error_code in pnids_errorcodes:
        error_summarizer.count(error_code, pnid)
    error_summarizer.summarize(total_pnids)
    if os.path.exists("errors/MODIFIED_MODEL_WARNING.txt"):
        with open("errors/MODIFIED_MODEL_WARNING.txt", "r") as f:
            model_number_errors = len(f.readlines())
            print(f"Be aware that {model_number_errors} files may be using a "
                  f"different model number than the one\nspecified by "
                  f"ProteinNet. See errors/MODIFIED_MODEL_WARNING.txt for "
                  f"a list of\nthese proteins.")


def write_errors_to_files(results_warnings, pnids):

    combined_data = {}

    # Define error dictionary for recording errors
    errors = {
        "failed": [],
        "single alignment, mask mismatch": [],
        "multiple alignments, mask mismatch": [],
        "multiple alignments, mask mismatch, many alignments": [],
        "multiple alignments, found matching mask": [],
        "multiple alignments, found matching mask, many alignments": [],
        "single alignment, mask mismatch, mismatch used in alignment": [],
        "multiple alignments, mask mismatch, mismatch used in alignment": [],
        "multiple alignments, mask mismatch, many alignments, mismatch used in "
        "alignment": [],
        "single alignment, found matching mask, mismatch used in alignment": [],
        "multiple alignments, found matching mask, mismatch used in alignment": [],
        "multiple alignments, found matching mask, many alignments, mismatch used in alignment":
            [],
        "mismatch used in alignment": [],
        "too many wrong AAs, mismatch used in alignment": [],
        'too many wrong AAs, multiple alignments, found matching mask, mismatch used in alignment':
            [],
        'bad gaps': [],
        'needs manual adjustment': []
    }

    # Delete/update ProteinNet entries depending on their ability to merge w/SidechainNet.
    for (combined_result, warning), pnid in zip(results_warnings, pnids):
        if combined_result:
            combined_data[pnid] = combined_result
        if warning:
            errors[warning].append(pnid)

    # Record ProteinNet IDs that could not be combined or exhibited warnings
    with open("errors/NEEDS_ADJUSTMENT.txt", "w") as f:
        for failed_id in errors["needs manual adjustment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_ERRORS.txt", "w") as f:
        for failed_id in errors["failed"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_1ALN_MISMATCH.txt", "w") as f:
        for failed_id in errors["single alignment, mask mismatch"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_M-ALN_MISMATCH.txt", "w") as f:
        for failed_id in errors["multiple alignments, mask mismatch"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_M-ALN_MISMATCH_MANY.txt", "w") as f:
        for failed_id in errors["multiple alignments, mask mismatch, many alignments"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_1ALN_MISMATCH_WRONGAA.txt", "w") as f:
        for failed_id in errors[
                "single alignment, mask mismatch, mismatch used in alignment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_M-ALN_MISMATCH_WRONGAA.txt", "w") as f:
        for failed_id in errors[
                "multiple alignments, mask mismatch, mismatch used in alignment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_M-ALN_MISMATCH_MANY_WRONGAA.txt", "w") as f:
        for failed_id in errors[
            "multiple alignments, mask mismatch, many alignments, mismatch used in " \
            "alignment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_1ALN_MATCH_WRONGAA.txt", "w") as f:
        for failed_id in errors[
                "single alignment, found matching mask, mismatch used in alignment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_M-ALN_MATCH_WRONGAA.txt", "w") as f:
        for failed_id in errors[
                "multiple alignments, found matching mask, mismatch used in alignment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_M-ALN_MATCH_MANY_WRONGAA.txt", "w") as f:
        for failed_id in errors[
            "multiple alignments, found matching mask, many alignments, mismatch used " \
            "in " \
            "alignment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_WRONGAA-only.txt", "w") as f:
        for failed_id in errors["mismatch used in alignment"]:
            f.write(f"{failed_id}\n")
    with open("errors/COMBINED_MANY-WRONGAA.txt", "w") as f:
        for failed_id in errors[
                'too many wrong AAs, multiple alignments, found matching mask, mismatch used in alignment']:
            f.write(f"{failed_id}\n")
    with open("errors/BAD_GAPS.txt", "w") as f:
        for failed_id in errors['bad gaps']:
            f.write(f"{failed_id}\n")

    return combined_data, errors
